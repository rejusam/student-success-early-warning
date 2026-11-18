variable "region" {
  description = "AWS region; the same as the foundation stack."
  type        = string
  default     = "ap-southeast-2"
}

variable "prefix" {
  description = "The same prefix as the foundation stack."
  type        = string
  default     = "ssew"
}

variable "image_uri" {
  description = "Image to run, pinned by digest: <repository url>@sha256:<digest>."
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
    error_message = "image_uri must end in @sha256:<digest>, so the deploy is pinned to one image."
  }
}

variable "memory_mb" {
  description = "Lambda memory. CPU is allocated in proportion."
  type        = number
  default     = 1024
}

variable "timeout_seconds" {
  description = "Lambda timeout. The HTTP API stops waiting at 30 seconds."
  type        = number
  default     = 30
}

variable "throttle_rate" {
  description = "Steady requests per second allowed by the HTTP API."
  type        = number
  default     = 5
}

variable "throttle_burst" {
  description = "Short burst of requests allowed by the HTTP API."
  type        = number
  default     = 10
}

variable "log_retention_days" {
  description = "Days to keep Lambda and API access logs."
  type        = number
  default     = 30
}
