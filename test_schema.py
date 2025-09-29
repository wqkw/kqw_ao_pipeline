from prompts.outputstruct_base_components import StoryboardSpec, DialogueLine
import json

# Test DialogueLine first
dialogue_schema = DialogueLine.model_json_schema()
print("DialogueLine schema:")
print(json.dumps(dialogue_schema, indent=2))
print("\nDialogueLine required fields:", dialogue_schema.get('required', []))
print("DialogueLine properties:", list(dialogue_schema.get('properties', {}).keys()))

print("\n" + "="*50 + "\n")

# Test full StoryboardSpec
storyboard_schema = StoryboardSpec.model_json_schema()
print("StoryboardSpec schema generated successfully")

# Look for any schemas with missing required fields
def find_schema_issues(obj, path=''):
    if isinstance(obj, dict):
        if 'properties' in obj and 'required' in obj:
            props = set(obj['properties'].keys())
            required = set(obj['required'])
            missing = props - required
            if missing:
                print(f'Schema at {path} missing required fields: {missing}')
                print(f'  Properties: {props}')
                print(f'  Required: {required}')
        for key, value in obj.items():
            find_schema_issues(value, f'{path}.{key}' if path else key)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_schema_issues(item, f'{path}[{i}]')

find_schema_issues(storyboard_schema)