# Service stack: the parts that change on every release (the Lambda function, its logs and the
# HTTP API). Applied by the deploy job in GitHub Actions with the deploy role from the
# foundation stack. State lives in the foundation's state bucket; the bucket, key and region
# are passed to `terraform init` with -backend-config (see .github/workflows/retrain.yml).
terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.70.0"
    }
  }

  backend "s3" {
    key = "service/terraform.tfstate"
    # Keep any workspace listing under service/, the only prefix the deploy role may list.
    workspace_key_prefix = "service/env"
    encrypt              = true
    use_lockfile         = true # S3 native locking, no DynamoDB table
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project    = "student-success-early-warning"
      stack      = "service"
      managed_by = "terraform"
    }
  }
}
