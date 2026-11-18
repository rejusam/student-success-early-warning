variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "ap-southeast-2"
}

variable "prefix" {
  description = "Short name put in front of every resource name."
  type        = string
  default     = "ssew"
}

variable "github_repository" {
  description = "owner/name of the GitHub repository allowed to use the roles."
  type        = string
  default     = "rejusam/student-success-early-warning"
}

variable "github_deploy_environment" {
  description = "GitHub environment whose jobs may assume the deploy role."
  type        = string
  default     = "production"
}

variable "github_score_environment" {
  description = "GitHub environment whose jobs may assume the score role."
  type        = string
  default     = "scoring"
}

variable "budget_email" {
  description = "Email address for budget alerts. Set in terraform.tfvars, which is not committed."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget in US dollars. Alerts at 80% actual and 100% forecast."
  type        = number
  default     = 10
}

variable "risk_list_retention_days" {
  description = "Days a risk list is kept in S3 before it is deleted."
  type        = number
  default     = 90
}

variable "allow_destroy" {
  description = "Set to true only to tear everything down: lets destroy delete non-empty buckets and the image repository (docs/deploy.md)."
  type        = bool
  default     = false
}
