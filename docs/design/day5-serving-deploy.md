# Day 5 design: serving, deployment and retraining

Date: 2026-11-18. Status: decisions below are open to review.

## Why
Days 1 to 4 built and checked a model. Day 5 puts it into use the way a university team would:
a weekly list for advisors, an API that other systems can call, a container that runs the same
way everywhere, a cloud deployment with no stored passwords, and a retraining pipeline that
runs the gate and the fairness audit before anything new is released. This is the part of the
work that turns a notebook result into a service.

## Goals
- Serve `lr_main`, the model in use (the gradient boosting challenger failed the gate,
  `reports/gate.json`).
- A batch job that writes, per module presentation and prediction week, the top 20% of
  students by risk with up to three reasons each.
- A FastAPI service that scores students sent to it, with strict input checks.
- One Docker image for both the API and the batch job: pinned base image, non-root user.
- Infrastructure on AWS written as code, deployed from GitHub Actions through OpenID Connect
  (OIDC) roles with only the permissions each job needs. No access keys anywhere.
- A retraining workflow that reruns the whole chain (data, baseline, gate, fairness audit,
  release) and only deploys a model that has passed through it.

## Non-goals
- Drift and live-accuracy monitoring (Day 6).
- A user interface for advisors. The list is a CSV in a private S3 bucket; a real deployment
  would feed it into the advisor system.
- Serving the gradient boosting model. It is not the champion.
- A shared MLflow server. The registry stays a file store; in CI it is kept as a workflow
  artifact.

## Anti-goals
- Any change to a model or to a number in `reports/`. Serving reads what Days 2 to 4 made.
  The release step checks that the files it ships reproduce `reports/baseline_metrics.json`
  exactly and stops if they do not.
- Demographic fields reaching the model through the API. The request schema rejects any field
  it does not know.
- Long-lived AWS keys, secrets in the repo, an open S3 bucket, or one role that can do
  everything.
- Using an AWS account that belongs to an employer or a client. The project gets its own
  account.

## Decisions
1. **`lr_main` is registered in MLflow as `champion`.** For each week, `lr_main` is fitted
   exactly as on Day 2, its 2014J metrics are checked against `reports/baseline_metrics.json`,
   and it is registered as a new version of `early-warning-week-<k>` with alias `champion` and
   tag `role=incumbent`. The gradient boosting versions keep alias `challenger`. The registry
   then answers "which model is in use?" by alias, which is how MLflow is meant to be used.
   If a later gradient boosting model passes the gate it would take the `champion` alias in
   `train_gbm.py`, and this step would leave it alone.
2. **The release step exports a model bundle; the service does not import MLflow.**
   `scripts/release_model.py` downloads each week's `champion` from the registry and writes
   `week-<k>.skops` files and a `manifest.json` (registry version and run, git commit, data
   checksum, feature list, scikit-learn version, SHA-256 of each file). It then loads the
   exported files, rescores 2014J and stops if any metric differs from
   `reports/baseline_metrics.json`. The service checks every file's SHA-256 against the
   manifest before loading it, and loads it with skops, which refuses types that are not on
   an allow-list (safer than pickle). Keeping MLflow out of the image makes it smaller and
   quicker to start.
3. **Only logistic regression champions can be released.** Reasons for `lr_main` come from its
   coefficients (decision 4). A gradient boosting champion would need SHAP in the image, so
   the release step refuses a champion that is not a logistic regression pipeline, with a
   message saying so, rather than shipping a model it cannot explain.
4. **Reasons from the logistic regression itself.** The model's log-odds are the intercept plus
   coefficient times scaled value for each input. Because the inputs are scaled to mean 0 on
   the training data, each term is that input's push on the risk compared with an average
   training student. A feature's push is the sum of its terms (value plus missing-value
   indicator). This is exact (a test checks that the pushes plus the intercept equal the
   model's log-odds) and is what SHAP's linear explainer gives for this model. The reasons
   are the three largest positive pushes, with the same plain-language labels as Day 3.
   The module is never a reason. The labels move from `reasons.py` to a small `labels.py`, so
   the service does not import SHAP.
5. **Batch job.** `scripts/score_batch.py --week K --presentation P` builds the week's cohort
   with the same SQL as training (`load_week`, so only data up to the cutoff day), scores it,
   takes the top 20% of each module presentation with `metrics.flag_top_k` (the list the
   metrics were computed on), and writes one CSV per module: rank, student id, risk and
   reasons. No outcome or demographic column is written. A `manifest.json` records the model
   versions and the counts.
6. **API.** FastAPI, two routes. `GET /health` returns status and the model versions.
   `POST /v1/score` takes a week (2, 4, 8 or 12) and 1 to 500 students, and returns each
   student's risk and reasons. Pydantic checks every field: known module codes only, counts
   not negative, average score from 0 to 100, unknown fields rejected, and consistency rules
   that hold in the training data (missed = due - submitted, late not above submitted, last
   week's clicks not above total clicks, registered by the cutoff day). A test sends every
   row of the test-sample cohort through the schema, so the rules cannot reject real rows.
   The API does not rank students: a top-20% list needs the whole module, which is the batch
   job's work.
7. **Dependencies split.** `mlflow` and `shap` move to a `train` dependency group that `uv sync`
   still installs by default. The image installs only the runtime dependencies (duckdb, pandas,
   scikit-learn, skops, fastapi, uvicorn) from the same `uv.lock`, so the scikit-learn version
   that trained the model is the one that loads it.
8. **Docker image.** Two stages. The first uses the uv image (pinned by digest) to install the
   locked runtime dependencies. The second is `python:3.12-slim-trixie` pinned by digest,
   with the virtual environment, `src/`, `sql/` and the model bundle copied in, bytecode
   compiled at build time, and a non-root user (uid 10001). The AWS Lambda Web Adapter
   (pinned by digest) is copied in as a Lambda extension, so the same image runs uvicorn
   locally, in CI and on Lambda with no code change. The batch job runs from the same image
   with a different command.
9. **AWS target: Lambda container behind an API Gateway HTTP API, in `ap-southeast-2`
   (Sydney).** Lambda costs nothing when idle and scales to zero, which fits a weekly list
   and occasional API calls. The HTTP API adds rate limits (5 requests a second, burst 10) so
   a public endpoint cannot run up a bill. App Runner and ECS Fargate were not chosen
   because they charge while idle. The region is a variable.
10. **Terraform, in two stacks.**
    - `infra/foundation/`: applied once by hand with an administrator login (AWS IAM Identity
      Center, short-lived). It creates the long-lived and sensitive parts: the GitHub OIDC
      provider, three IAM roles, the ECR repository (immutable tags, scan on push, keep the
      last 10 images), the S3 bucket for Terraform state, the S3 bucket for risk lists
      (private, encrypted, TLS only, versioned, objects deleted after 90 days) and a monthly
      budget alarm. State is a local file that is not committed.
    - `infra/service/`: applied by GitHub Actions on every deploy. The Lambda function, its log
      group (30-day retention) and the HTTP API. State in the S3 bucket with S3 native
      locking.
    Keeping roles and buckets out of the service stack means the deploy role cannot create
    roles or delete data.
11. **Three roles, each for one job.** The two GitHub roles trust only this repository, each
    in one GitHub environment (OIDC subject
    `repo:rejusam/student-success-early-warning:environment:<name>`; see change 6 below).
    - `deploy`: push to the one ECR repository, read and write the service state file, manage
      the one Lambda function, its log group and HTTP APIs, and pass the one Lambda role.
    - `score`: pull from the ECR repository, read the Lambda function's image address, and
      write objects under `risk-lists/` in the risk-list bucket.
    - `lambda`: the function's own role, which can only write its logs.
    API Gateway permissions cannot be limited to one API by name, so the deploy role is
    limited to API Gateway in the one region.
12. **Workflows.**
    - `ci.yml` (every push and pull request) gains two jobs: `container` builds the image with
      a model bundle made from the test sample, runs it, calls `/health` and `/v1/score`, and
      scans it with Trivy (fails on fixable HIGH or CRITICAL); `infra` runs `terraform fmt`,
      `terraform validate` and a Trivy misconfiguration scan on both stacks.
    - `retrain.yml` (manual): `train` job downloads OULAD (checksum checked), ingests, runs the
      data checks, the baseline, gradient boosting tuning and test with the gate, the fairness
      audit and the release step, then checks that `reports/` is unchanged (`git diff
      --exit-code`), and uploads the bundle, the reports and the MLflow store as workflow
      artifacts. A failed gate is a normal result (the champion stays), so the workflow goes
      on; any other error stops it. `deploy` job (only when the AWS variables are set) builds
      the image with that bundle, scans it, pushes it to ECR, applies `infra/service` with the
      image digest and calls the live `/health` and `/v1/score`. Then it calls `score.yml`.
    - `score.yml` (manual, or called by `retrain.yml`): prepares the week's data, pulls the
      image that Lambda is running, runs the batch job in it and copies the lists to S3.
    All three are manual because OULAD does not change. In use, `score.yml` would run weekly
    on a schedule and `retrain.yml` once per teaching period; each is a one-line change.
    Every action is pinned by commit SHA and every image by digest, as in `ci.yml`.
13. **Retraining rescores 2014J with fixed settings.** Each run of `retrain.yml` repeats the
    Day 2 to 4 runs on the same data with the same settings. That is a reproducibility check,
    not a new look at the test set: nothing is chosen from it, and the run fails if any report
    changes. With new data, the train and test presentations would move forward.
14. **What a person must do.** Create a new AWS account for this project (not an employer's or
    client's), turn on MFA and a budget, apply `infra/foundation` once, set six GitHub
    variables and the `production` and `scoring` environments. Steps in `docs/deploy.md`. Until then,
    `retrain.yml` trains and releases but skips the deploy, and CI checks everything that does
    not need an account.

## Approaches considered
1. **Lambda container + HTTP API + Terraform, batch in the same image run by GitHub Actions
   (chosen).** No cost when idle, one image for both uses, and an IaC tool most teams know.
2. **ECS Fargate service + scheduled task, CDK.** Closer to a long-running service, but it
   charges every hour, and CDK adds a Node.js toolchain to a Python project.
3. **SageMaker endpoint + batch transform.** Built for this, but an endpoint costs money while
   idle and the setup is harder to explain line by line for a logistic regression.

## Outputs
| Path | Contents |
|---|---|
| `src/early_warning/{labels,explain,release,scoring,api,batch}.py` | label text, linear reasons, release step, bundle loading and scoring, API, batch job |
| `scripts/release_model.py`, `scripts/score_batch.py` | command-line entry points |
| `Dockerfile`, `.dockerignore` | the image |
| `infra/foundation/`, `infra/service/` | Terraform stacks |
| `.github/workflows/{ci,retrain,score}.yml` | CI, retraining and deployment, batch scoring |
| `docs/deploy.md` | account setup, first deploy, teardown |
| `reports/` | unchanged |

## Tests
- Reasons: pushes plus intercept equal the log-odds; module never a reason; only positive
  pushes; a missing score gives "no assessment score yet".
- Release (test sample): registers `lr_main` as `champion`, a second run adds no version, the
  bundle's hashes match, a non-logistic champion is refused, smoke mode refuses `reports/`
  and `mlruns/`.
- Scoring: a changed bundle file is refused; each module list has ceil(0.2 n) rows; no
  outcome or demographic column in the output.
- API: valid request; unknown field (for example `gender`) rejected; inconsistent counts
  rejected; week 3 rejected; more than 500 students rejected; every test-sample row accepted,
  and API risks equal the pipeline's own probabilities.

## Constraints
- Same as before: Python 3.12, `uv.lock`, code simple enough to explain line by line, every
  number in the docs from `reports/`, commit and push only when the owner says so.
- Release versions at least a week old for actions, images and providers (Terraform 1.17.1,
  AWS provider 6.70.x, uv 0.11.5 image, Lambda Web Adapter 1.1.0, Trivy 0.76.2).

## Left to the builder
- JSON layout of the manifest, function names, test layout.
- Lambda memory and timeout (start at 1,024 MB and 30 seconds).

## Changes during the build
1. **Grouped reasons.** The first risk lists gave reasons such as "assessments submitted so far:
   1" and "clicks on the course site so far: 1005" as pushes up the risk. Both are exact, and
   both come from inputs that move together: missed is always due minus submitted, and the
   five activity counts are strongly related, so the logistic regression can give one of them
   a small positive weight while the others carry a negative one (at week 8 `clicks_total`
   +0.099, `weeks_active` -0.4179, `reports/baseline_coefficients.csv`). The pushes of each
   group are now added up into one reason ("submitted 1 of 2 assessments due so far";
   "course-site activity so far: ..."). The total is still exact; only the split inside a
   group, which the model cannot pin down, is no longer shown. No model changed.
2. **Six GitHub variables, not four:** region, the two role ARNs, the state bucket, the ECR
   repository URL and the risk-list bucket.
3. **Buckets as a small module** (`infra/modules/private-bucket`). Written with `for_each` over
   a map, the Trivy scan could not see the public-access blocks and reported them missing. A
   module makes every control explicit per bucket. Accepted findings, each with its reason,
   are in `.trivyignore`.
4. **`allow_destroy`** (default false) on the foundation stack lets teardown delete the
   versioned buckets and the image repository without emptying them by hand.
5. **`python -m early_warning.batch`** needed a `__main__` block to run from the image.
6. **Review fixes.** Separate GitHub environments per role (`production` for deploy,
   `scoring` for the risk lists), so a scoring job cannot assume the deploy role; a Deny on
   Lambda function URLs (a second public way in that would skip the rate limits); Terraform
   workspace listings kept under the `service/` prefix the deploy role may list; upper bounds
   on API counts, so a huge number gives a 422 and not a server error.
7. **Two Day 3 reports regenerated.** The local reproducibility run of the whole retrain chain
   (fresh MLflow store) reproduced every report exactly except the week-4 calibration bins in
   `gbm_calibration.json` and `gbm_validation.json` (ECE 0.0592 to 0.059 uncalibrated, 0.0551
   to 0.0549 calibrated). Cause: tied probabilities are split into bins by row order, and the
   feature query got its fixed `ORDER BY` on Day 4, after those files were written on Day 3.
   The regenerated files are what the committed code produces; keeping the old ones would make
   `retrain.yml` fail on every run. No model, metric, gate result or audit number changed.
