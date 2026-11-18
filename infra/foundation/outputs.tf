# Set these as GitHub repository variables (docs/deploy.md). They are not secrets.
output "AWS_REGION" {
  value = var.region
}

output "AWS_DEPLOY_ROLE_ARN" {
  value = aws_iam_role.deploy.arn
}

output "AWS_SCORE_ROLE_ARN" {
  value = aws_iam_role.score.arn
}

output "TF_STATE_BUCKET" {
  value = module.state_bucket.id
}

output "ECR_REPOSITORY_URL" {
  value = aws_ecr_repository.api.repository_url
}

output "RISK_LIST_BUCKET" {
  value = module.risk_list_bucket.id
}
