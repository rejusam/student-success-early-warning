# Deploying to AWS

How to set up a new AWS account for this project, deploy the scoring service from GitHub
Actions, and tear it all down. Design: [`docs/design/day5-serving-deploy.md`](design/day5-serving-deploy.md).

What gets created:

| Stack | Applied by | Contents |
|---|---|---|
| `infra/foundation/` | you, once, with an administrator sign-in | GitHub OIDC provider; roles `ssew-github-deploy`, `ssew-github-score`, `ssew-lambda`; ECR repository `ssew-api`; S3 buckets for Terraform state and risk lists; monthly budget alarm |
| `infra/service/` | the `deploy` job in `.github/workflows/retrain.yml` | Lambda function `ssew-api` (container image), its log group, an HTTP API with two routes and rate limits, the API access log group |

No AWS access keys are created or stored. GitHub Actions gets short-lived credentials through
OpenID Connect. The deploy role trusts only jobs in this repository's `production`
environment, and the score role only jobs in its `scoring` environment.

## 1. A new AWS account
Use an account that belongs to this project only, not one owned by an employer or a client.

1. Sign up at https://aws.amazon.com/ and read the free-tier terms that apply to new accounts
   at the time (https://aws.amazon.com/free/, https://aws.amazon.com/free/terms/). Accounts
   opened since 15 July 2025 choose a Free plan or a Paid plan; the Free plan closes the
   account after 6 months or when the credits run out.
2. Turn on MFA for the root user. Do not create access keys for the root user.
3. On the Free plan, do not turn on AWS Organizations, Control Tower or IAM Identity Center:
   the free-tier terms move an account that joins an organization to the Paid plan, and
   Identity Center creates one. Instead, as root, create an IAM user `ssew-admin` with
   console access, MFA and the `AdministratorAccess` policy, and no access keys. Sign in as
   that user from now on.
4. Sign in from the terminal with a named profile (AWS CLI 2.32 or newer; `aws login` gives
   short-lived credentials from the console sign-in, so no access keys are needed), and check
   you are in the new account:
   ```bash
   aws login --profile ssew-admin   # region ap-southeast-2; opens the browser
   aws sts get-caller-identity --profile ssew-admin   # the Account must be the new one
   ```
   On the Paid plan, IAM Identity Center with `aws configure sso` is the better choice.

## 2. Apply the foundation stack (once)
Needs Terraform 1.17 (`brew install hashicorp/tap/terraform`, or the binary from
https://developer.hashicorp.com/terraform/install).

```bash
cd infra/foundation
cp terraform.tfvars.example terraform.tfvars   # set budget_email; the file is not committed
export AWS_PROFILE=ssew-admin
terraform init
terraform plan -out plan.tfplan     # read it: about 25 resources, nothing outside this project
terraform apply plan.tfplan
terraform output
```

The state file (`terraform.tfstate`) stays on your machine and is gitignored. Keep a copy
somewhere safe; without it, later changes to this stack need `terraform import`.

## 3. Connect GitHub
1. In the repository settings, create two environments, each allowing deployments from `main`
   only: `production` (the deploy job) and `scoring` (the risk-list job). Each AWS role trusts
   one of them, so the scoring job cannot assume the deploy role. Adding yourself as a
   required reviewer on `production` makes every deploy wait for your approval.
2. Set the stack outputs as repository variables (they are not secrets):
   ```bash
   for name in AWS_REGION AWS_DEPLOY_ROLE_ARN AWS_SCORE_ROLE_ARN TF_STATE_BUCKET \
               ECR_REPOSITORY_URL RISK_LIST_BUCKET; do
     gh variable set "$name" --body "$(terraform output -raw "$name")"
   done
   ```

Until `AWS_DEPLOY_ROLE_ARN` is set, `retrain.yml` trains and releases the model and skips the
deploy and scoring jobs.

## 4. Deploy
```bash
gh workflow run retrain.yml -f week=12 -f presentation=2014J
gh run watch
```

The run retrains and checks everything (about 30 to 60 minutes), builds and scans the image,
pushes it to ECR, applies `infra/service` with the image digest, calls the live API, and
writes the risk lists to S3. The live API address is in the run summary. To score another week
with the model already deployed: `gh workflow run score.yml -f week=8 -f presentation=2014J`.

Call the live API:
```bash
curl -s "$API_URL/health"
curl -s -X POST "$API_URL/v1/score" -H 'Content-Type: application/json' \
  -d @docs/examples/score_request.json
```

Read a risk list (with the administrator profile):
```bash
aws s3 cp "s3://$(terraform -chdir=infra/foundation output -raw RISK_LIST_BUCKET)/risk-lists/2014J/week-12/AAA.csv" -
```

## If the first deploy fails with AccessDenied
The deploy role lists only the actions this project needs. The list was written from the AWS
documentation and has not yet been run against a live account. If a step fails with
AccessDenied, the error names the missing action: add it to the matching statement in
`infra/foundation/iam.tf`, run `terraform apply` there again, and rerun the job.

## Cost
Idle cost is close to zero: Lambda and the HTTP API charge per request, ECR and S3 per GB
stored (the lifecycle rule keeps the last 10 images). The HTTP API allows 5
requests a second (bursts of 10), which caps what a stranger calling the public URL can cost.
The budget alarm emails at 80% of the monthly budget (default 10 USD) and when the forecast
passes 100%. It counts usage before credits, so it still warns while free-tier credits are
paying. Prices: https://aws.amazon.com/lambda/pricing/,
https://aws.amazon.com/api-gateway/pricing/, https://aws.amazon.com/ecr/pricing/.

## Tear down
```bash
export AWS_PROFILE=ssew-admin

# 1. The service stack (its state is in the S3 bucket)
cd infra/service
terraform init -backend-config="bucket=$(terraform -chdir=../foundation output -raw TF_STATE_BUCKET)" \
  -backend-config="region=ap-southeast-2"
terraform destroy -var "image_uri=unused@sha256:$(printf '0%.0s' {1..64})"

# 2. The foundation stack. allow_destroy lets Terraform delete the buckets and the image
#    repository even though they are not empty.
cd ../foundation
terraform apply -var allow_destroy=true
terraform destroy -var allow_destroy=true
```

Then delete the GitHub variables and the `production` environment, and close the AWS account
if it is no longer needed.
