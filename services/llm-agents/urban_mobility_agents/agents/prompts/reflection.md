# TASK INSTRUCTION
Reflect on past experiences to identify patterns, lessons, and insights that will improve future travel planning.
You can also generate concepts about important things or ideas to be remembered in the long term memory (optional but recommended).

# OUTPUT FORMAT
Must be in the json format
```json
{
    "reflection": "string - narrative reflection on the day",
    "concepts": [["<content>", "<keywords>", "<spatial_scope>", "<time_scope>"], ...]
}
```

# REFLECTION GUIDELINES
"reflection" is a string that summaries what's happened today. Be specific when mention a trip, that includes this tag in the context [**CURRENT TIME**: <current_time>, **TRAVEL TO**: <travel_to>]:
- **CURRENT TIME**: The current time of the travel.
- **TRAVEL TO**: The destination of the travel.
Keep it under 200 words.

### Structure your reflection around:
1. **Plan Adherence**
- Which activities went as planned?
- What required adjustment and why?
- Were time estimates accurate?
- Were the arrival times as expected?

2. **Traffic & Transportation**
- Route performance vs. expectations
- Travel mode effectiveness

3. **Key Learnings**
- What would you do differently?
- Patterns worth remembering
- Planning improvements for tomorrow
- Planning worth keeping in future

### CONCEPTS GUIDELINES
"concepts" a list of concepts to be remembered (optional but recommended). These memories will help you to make better future decisions.

Generate 0-5 concepts for long-term memory. Focus on:
- Recurring patterns (not one-time events); especially traffic related patterns
- Actionable insights for future planning
- Specific locations and time windows

**Content Types:**
- Traffic patterns: "Bus 69 travel time is reliable"
- Activity timing: "Wait long time for Bus 20"
DO NOT fabricate concepts; base patterns on your actual experiences.

**Required Format for each concept:**
["<content>", "<keywords>", "<spatial_scope>", "<time_scope>", "<purpose>"]

**Spatial Scope Options:**
- Specific Transport Mode and Route: e.g. "Bus 69"
- Facility: e.g. "Stop (Jean Jaures)"

# INPUT GUIDELINES
Now you will be provided with a list of past experiences and observations. Use these to generate your reflection and concepts following the above guidelines.

```json
$experiences_text
```
$custom_guidelines