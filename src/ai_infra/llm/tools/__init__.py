from ai_infra.llm.shell.middleware import ShellMiddleware, ShellMiddlewareConfig
from ai_infra.llm.shell.tool import create_shell_tool, run_shell
from ai_infra.llm.tools.approval import (
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalRule,
    AsyncApprovalHandler,
    AsyncOutputReviewer,
    MultiApprovalRequest,
    OutputReviewer,
    OutputReviewRequest,
    OutputReviewResponse,
    auto_approve_handler,
    auto_reject_handler,
    console_approval_handler,
    create_rule_based_handler,
    create_selective_handler,
)
from ai_infra.llm.tools.custom.multimodal import (
    analyze_image,
    generate_image,
    transcribe_audio,
)
from ai_infra.llm.tools.custom.web import fetch_url, search_web
from ai_infra.llm.tools.events import ApprovalEvent, ApprovalEvents
from ai_infra.llm.tools.hitl import (
    ApprovalConfig,
    HITLConfig,
    ToolExecutionConfig,
    ToolExecutionError,
    ToolPolicy,
    ToolTimeoutError,
    ToolValidationError,
    apply_output_gate,
    apply_output_gate_async,
    compute_effective_tools,
    maybe_await,
    wrap_tool_for_approval,
    wrap_tool_for_hitl,
    wrap_tool_with_execution_config,
)
from ai_infra.llm.tools.tool_controls import ToolCallControls
from ai_infra.llm.tools.tools import tools_from_functions

__all__ = [
    # Approval
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalRule",
    "AsyncApprovalHandler",
    "AsyncOutputReviewer",
    "MultiApprovalRequest",
    "OutputReviewRequest",
    "OutputReviewResponse",
    "OutputReviewer",
    "auto_approve_handler",
    "auto_reject_handler",
    "console_approval_handler",
    "create_rule_based_handler",
    "create_selective_handler",
    # Events
    "ApprovalEvent",
    "ApprovalEvents",
    # HITL
    "ApprovalConfig",
    "HITLConfig",
    "ToolExecutionConfig",
    "ToolExecutionError",
    "ToolPolicy",
    "ToolTimeoutError",
    "ToolValidationError",
    "apply_output_gate",
    "apply_output_gate_async",
    "compute_effective_tools",
    "maybe_await",
    "wrap_tool_for_approval",
    "wrap_tool_for_hitl",
    "wrap_tool_with_execution_config",
    # Tool controls
    "ToolCallControls",
    # Tools
    "tools_from_functions",
    # Multimodal tools
    "transcribe_audio",
    "analyze_image",
    "generate_image",
    "fetch_url",
    "search_web",
    # Shell tools
    "run_shell",
    "create_shell_tool",
    # Shell middleware
    "ShellMiddleware",
    "ShellMiddlewareConfig",
]
