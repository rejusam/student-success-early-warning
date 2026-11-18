data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  function_name = "${var.prefix}-api"
  # The role is made by the foundation stack, so the deploy role never creates IAM roles.
  lambda_role_arn = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${var.prefix}-lambda"
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "api" {
  function_name = local.function_name
  description   = "Student early-warning scoring API (FastAPI through the Lambda Web Adapter)"
  package_type  = "Image"
  image_uri     = var.image_uri
  role          = local.lambda_role_arn
  architectures = ["x86_64"]
  memory_size   = var.memory_mb
  timeout       = var.timeout_seconds

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.lambda.name
  }
}

resource "aws_apigatewayv2_api" "api" {
  name          = "${var.prefix}-api"
  protocol_type = "HTTP"
  description   = "Public routes of the scoring API, rate limited"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

# Only these two routes are open. Anything else, including the interactive docs, gets a 404
# from API Gateway without reaching Lambda.
resource "aws_apigatewayv2_route" "routes" {
  for_each  = toset(["GET /health", "POST /v1/score"])
  api_id    = aws_apigatewayv2_api.api.id
  route_key = each.value
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/apigateway/${var.prefix}-api"
  retention_in_days = var.log_retention_days
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = var.throttle_rate
    throttling_burst_limit = var.throttle_burst
  }

  # One line per request: no request body, so no student features are logged.
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format = jsonencode({
      requestId = "$context.requestId"
      time      = "$context.requestTime"
      route     = "$context.routeKey"
      status    = "$context.status"
      latencyMs = "$context.integrationLatency"
    })
  }
}

resource "aws_lambda_permission" "api" {
  statement_id  = "AllowHttpApi"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
