You decide whether one agent message asks the user about a standing rule.

A standing rule is a user preference, habit, or default that would shape
the agent's behavior on a different future task.

## Input

- `<reason>`: the agent's free-form intent for this turn. Treat it as
  the agent's own statement of why it is speaking.
- `<output>`: what the agent actually said to the user.

## Decision

Return `is_strict_rule_question = true` when `<output>` contains a question
the agent itself asks about a standing rule. Use `<reason>` as supporting
evidence, but do not require it to explicitly say "standing rule" if the
user-visible output clearly asks for a future/default behavior.

Return `false` when `<reason>` or `<output>` only applies, restates, or
confirms a rule the user has already stated.

Recall of past rules, status updates, and quoted draft content are
context, not asks.

Return `false` in any other case.

## rule_question_span

When `true`, copy one complete verbatim standing-rule ask sentence from
`<output>`. Direct questions such as `would you ...?` and soft/indirect asks
such as `let me know if you'd like ...` both count when they ask about a
future/default behavior. Include sentence-level lead-ins or framing that belong
to that ask, such as `for future ...`, `For recurring ...`, `Quick question for
the future:`, or `Since ..., would you ...`. Do not paraphrase or include
previous independent status, recall, draft, or current-task sentences. Preserve
the original capitalization, quotes, punctuation, and line breaks exactly.

When `false`, set to `null`.

{{FEW_SHOT_EXAMPLES}}

## Output format

Return exactly one JSON object, nothing else:

```json
{
  "is_strict_rule_question": false,
  "rule_question_span": null
}
```
