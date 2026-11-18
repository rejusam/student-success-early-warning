data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id    = data.aws_caller_identity.current.account_id
  partition     = data.aws_partition.current.partition
  function_name = "${var.prefix}-api"
  function_arn  = "arn:${local.partition}:lambda:${var.region}:${local.account_id}:function:${local.function_name}"
  # The service stack creates these two log groups; the roles below may use only them.
  lambda_log_group = "/aws/lambda/${local.function_name}"
  api_log_group    = "/aws/apigateway/${var.prefix}-api"
}

# --- Container registry -----------------------------------------------------------------

resource "aws_ecr_repository" "api" {
  name                 = "${var.prefix}-api"
  image_tag_mutability = "IMMUTABLE" # a tag always means the same image
  force_delete         = var.allow_destroy

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# Lambda pulls the image with its own service identity, and only for our one function.
data "aws_iam_policy_document" "ecr_lambda_pull" {
  statement {
    sid     = "LambdaPull"
    actions = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "aws:sourceArn"
      values   = ["${local.function_arn}*"] # the function and its versions
    }
  }
}

resource "aws_ecr_repository_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy     = data.aws_iam_policy_document.ecr_lambda_pull.json
}

# --- S3 buckets: Terraform state for the service stack, and the weekly risk lists --------

module "state_bucket" {
  source        = "../modules/private-bucket"
  name          = "${var.prefix}-tfstate-${local.account_id}"
  force_destroy = var.allow_destroy
}

module "risk_list_bucket" {
  source        = "../modules/private-bucket"
  name          = "${var.prefix}-risk-lists-${local.account_id}"
  force_destroy = var.allow_destroy
}

resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = module.state_bucket.id
  rule {
    id     = "old-state-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# Risk lists name students, so they are kept only as long as an advisor needs them.
resource "aws_s3_bucket_lifecycle_configuration" "risk_lists" {
  bucket = module.risk_list_bucket.id
  rule {
    id     = "expire-risk-lists"
    status = "Enabled"
    filter {}
    expiration {
      days = var.risk_list_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# --- Budget alarm --------------------------------------------------------------------------

resource "aws_budgets_budget" "monthly" {
  name         = "${var.prefix}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Count usage before credits, so the alarm still fires while free-tier credits pay the bill.
  cost_types {
    include_credit = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
