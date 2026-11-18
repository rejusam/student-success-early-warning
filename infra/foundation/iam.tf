# GitHub Actions signs in with OpenID Connect: each job gets a short-lived token from GitHub,
# and AWS swaps it for temporary credentials of one role. No access keys are stored anywhere.
resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

# Each GitHub role trusts one GitHub environment of this repository: deploy jobs run in
# `production`, the scoring job in `scoring`. So a scoring job cannot assume the deploy role.
data "aws_iam_policy_document" "github_trust" {
  for_each = {
    deploy = var.github_deploy_environment
    score  = var.github_score_environment
  }
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${each.value}"]
    }
  }
}

# --- Lambda's own role: it can only write its logs ----------------------------------------

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.prefix}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid       = "WriteOwnLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:${local.lambda_log_group}:*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "logs"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

# --- Deploy role: push the image and apply the service stack ------------------------------

resource "aws_iam_role" "deploy" {
  name                 = "${var.prefix}-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_trust["deploy"].json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # this action has no resource to limit it to
  }
  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.api.arn]
  }
  statement {
    sid       = "StateList"
    actions   = ["s3:ListBucket"]
    resources = [module.state_bucket.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["service/*"]
    }
  }
  statement {
    sid       = "StateReadWrite" # the state file and its lock file
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${module.state_bucket.arn}/service/*"]
  }
  statement {
    sid       = "ManageTheFunction"
    actions   = ["lambda:*"]
    resources = [local.function_arn, "${local.function_arn}:*"]
  }
  statement {
    # A function URL would be a second public way in that skips the API's rate limits.
    sid       = "NoFunctionUrl"
    effect    = "Deny"
    actions   = ["lambda:CreateFunctionUrlConfig", "lambda:UpdateFunctionUrlConfig"]
    resources = ["*"]
  }
  statement {
    sid       = "PassOnlyTheLambdaRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.lambda.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["lambda.amazonaws.com"]
    }
  }
  statement {
    sid     = "ManageTheLogGroups"
    actions = ["logs:*"]
    resources = [
      "arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:${local.lambda_log_group}",
      "arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:${local.lambda_log_group}:*",
      "arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:${local.api_log_group}",
      "arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:${local.api_log_group}:*",
    ]
  }
  statement {
    # Needed to read log groups and to turn on API access logging. These actions have no
    # resource to limit them to.
    sid = "LogsAccountLevel"
    actions = [
      "logs:CreateLogDelivery",
      "logs:DeleteLogDelivery",
      "logs:DescribeLogGroups",
      "logs:DescribeResourcePolicies",
      "logs:GetLogDelivery",
      "logs:ListLogDeliveries",
      "logs:PutResourcePolicy",
      "logs:UpdateLogDelivery",
    ]
    resources = ["*"]
  }
  statement {
    # API Gateway permissions cannot name one API before it exists, so they are limited to
    # HTTP APIs in this region.
    sid     = "ManageHttpApis"
    actions = ["apigateway:DELETE", "apigateway:GET", "apigateway:PATCH", "apigateway:POST", "apigateway:PUT", "apigateway:TagResource", "apigateway:UntagResource"]
    resources = [
      "arn:${local.partition}:apigateway:${var.region}::/apis",
      "arn:${local.partition}:apigateway:${var.region}::/apis/*",
      "arn:${local.partition}:apigateway:${var.region}::/tags/*",
    ]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

# --- Score role: run the live image and write the risk lists ------------------------------

resource "aws_iam_role" "score" {
  name                 = "${var.prefix}-github-score"
  assume_role_policy   = data.aws_iam_policy_document.github_trust["score"].json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "score" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # this action has no resource to limit it to
  }
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    resources = [aws_ecr_repository.api.arn]
  }
  statement {
    sid       = "FindTheLiveImage"
    actions   = ["lambda:GetFunction"]
    resources = [local.function_arn]
  }
  statement {
    sid       = "WriteRiskLists"
    actions   = ["s3:PutObject"]
    resources = ["${module.risk_list_bucket.arn}/risk-lists/*"]
  }
}

resource "aws_iam_role_policy" "score" {
  name   = "score"
  role   = aws_iam_role.score.id
  policy = data.aws_iam_policy_document.score.json
}
