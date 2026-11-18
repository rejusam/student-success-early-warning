# Foundation stack: the long-lived and sensitive parts (identity, roles, buckets, registry,
# budget). Applied once, by hand, by a person signed in with short-lived administrator
# credentials (IAM Identity Center). Its state is a local file that is never committed.
terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.70.0"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project    = "student-success-early-warning"
      stack      = "foundation"
      managed_by = "terraform"
    }
  }
}
