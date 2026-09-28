from string import Template

from evalbench.schemas import Category, Urgency

CATEGORIES: dict[Category, str] = {
    "billing": "Payments, invoices, charges, refunds, subscription plans and pricing",
    "bug": "The product does not work as expected (errors, crashes, wrong data, broken UI)",
    "account": "Login, authentication, users and permissions, account data and deletion",
    "feature_request": "A suggestion or request for a capability the product does not have yet",
    "other": "Anything else: thanks, press, jobs, partnerships, off-topic",
}

# order matters (used as a scale by Jev)
URGENCY_LEVELS: dict[Urgency, str] = {
    "low": "Question, suggestion or minor annoyance; no real impact on the customer's work",
    "medium": "A real problem or a deadline within days, but the customer can keep working",
    "high": "Customer is blocked, is losing money, or faces a security risk; needs action today",
}

REFUND_DEFINITION = "The customer explicitly asks to be refunded or reimbursed"


def render_prompt(template: str) -> str:
    return Template(template).safe_substitute(
        categories=_bullets(CATEGORIES),
        urgency_levels=_bullets(URGENCY_LEVELS),
        refund=REFUND_DEFINITION,
    )


def _bullets(definitions: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {description}" for name, description in definitions.items())
