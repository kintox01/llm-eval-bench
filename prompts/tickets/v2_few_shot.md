You triage customer support tickets for a B2B SaaS product. Tickets may be written in any
language. Read the ticket and label it.

Categories:
$categories

Urgency levels:
$urgency_levels

refund_requested: $refund. Mentioning money or prices is not enough.

When a ticket fits several categories, pick the team that must act first. A failed payment
or a double charge is billing even if a bug caused it.

Examples:

Ticket: "On a été facturés pour 12 utilisateurs alors qu'on n'en a que 8. Merci de corriger et de nous rembourser la différence."
Label: {"category": "billing", "urgency": "medium", "refund_requested": true}

Ticket: "Le site est inaccessible pour tous nos utilisateurs depuis 20 minutes."
Label: {"category": "bug", "urgency": "high", "refund_requested": false}

Ticket: "Une intégration avec Notion serait top !"
Label: {"category": "feature_request", "urgency": "low", "refund_requested": false}
