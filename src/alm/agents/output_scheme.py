from typing import Literal

from pydantic import BaseModel, Field


# create stractued output for summary log and categorize log
class SummarySchema(BaseModel):
    summary: str = Field(description="Summary of the log")


class ClassifySchema(BaseModel):
    category: Literal[
        "Cloud Infrastructure / AWS Engineers",
        "Kubernetes / OpenShift Cluster Admins",
        "DevOps / CI/CD Engineers (Ansible + Automation Platform)",
        "Networking / Security Engineers",
        "System Administrators / OS Engineers",
        "Application Developers / GitOps / Platform Engineers",
        "Identity & Access Management (IAM) Engineers",
        "Other / Miscellaneous",
    ] = Field(description="Category of the log")


class RouterStepByStepSolutionSchema(BaseModel):
    suggestion: Literal["No More Context Needed", "Need More Context"] = Field(
        description="The suggestion for the step by step solution: 'No More Context Needed' if the solution is straightforward, 'Need More Context' if the solution is complex"
    )
