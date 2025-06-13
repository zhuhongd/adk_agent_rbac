This is just an alternative version of the agent. I'm keeping it here in case I want to revisit it later.

While developing this version, we ran into some issues with how the root agent delegated tasks to the sub-agents. Initially, the root agent correctly passed the first few questions to the appropriate sub-agents. But after a few exchanges, the sub-agents either didn't return to the root agent or responded with generic fallback messages like “I don’t know” or “I’m not able to help with that.”

We weren’t sure how to fix this delegation flow, so in the official version, we ended up merging all the sub-agents into a single larger sub-agent and gave it access to all the tools used by the others. That way, we could ensure a more consistent and controlled response behavior.