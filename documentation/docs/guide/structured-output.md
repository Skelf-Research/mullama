# Structured Output

Generate output that conforms to a JSON Schema, ensuring the model produces valid, parseable structured data. This is essential for applications that need to extract information, generate API responses, or produce machine-readable output.

## Overview

Structured output works by converting a JSON Schema into a GBNF grammar that constrains the model's output. The model can only produce tokens that form valid JSON matching your schema, guaranteeing parseable results every time.

## JSON Schema-Based Generation

Define your desired output structure as a JSON Schema:

=== "Node.js"

    ```javascript
    import { Model, Context, StructuredOutput } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    const schema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'integer' },
        email: { type: 'string' },
      },
      required: ['name', 'age'],
    };

    const response = await context.generate(
      "Extract user info: John Doe is 30 years old, john@example.com",
      200,
      { schema }
    );

    const data = JSON.parse(response);
    console.log(data.name);  // "John Doe"
    console.log(data.age);   // 30
    ```

=== "Python"

    ```python
    import json
    from mullama import Model, Context, StructuredOutput

    model = Model.load("./model.gguf")
    context = Context(model)

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
        "required": ["name", "age"],
    }

    response = context.generate(
        "Extract user info: John Doe is 30 years old, john@example.com",
        max_tokens=200,
        schema=schema,
    )

    data = json.loads(response)
    print(data["name"])   # "John Doe"
    print(data["age"])    # 30
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams, StructuredOutput};
    use serde_json::Value;

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "email": { "type": "string" }
        },
        "required": ["name", "age"]
    });

    let structured = StructuredOutput::from_schema(&schema)?;
    let response = context.generate_with_grammar(
        "Extract user info: John Doe is 30 years old, john@example.com",
        200,
        &structured.grammar()
    )?;

    let data: Value = serde_json::from_str(&response)?;
    println!("Name: {}", data["name"]);
    println!("Age: {}", data["age"]);
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b \
      "Extract user info: John Doe is 30 years old, john@example.com" \
      --schema '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}'
    ```

## Converting JSON Schema to Grammar

Under the hood, Mullama converts JSON Schemas to GBNF grammars using `JsonSchemaConverter`:

=== "Node.js"

    ```javascript
    import { JsonSchemaConverter } from 'mullama';

    const schema = {
      type: 'object',
      properties: {
        title: { type: 'string' },
        score: { type: 'number', minimum: 0, maximum: 100 },
      },
      required: ['title', 'score'],
    };

    // Convert schema to grammar string
    const grammar = JsonSchemaConverter.convert(schema);
    console.log(grammar);  // GBNF grammar string

    // Use grammar directly
    const response = await context.generate("Rate this movie:", 200, { grammar });
    ```

=== "Python"

    ```python
    from mullama import JsonSchemaConverter

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": "number", "minimum": 0, "maximum": 100},
        },
        "required": ["title", "score"],
    }

    # Convert schema to grammar string
    grammar = JsonSchemaConverter.convert(schema)
    print(grammar)  # GBNF grammar string

    # Use grammar directly
    response = context.generate("Rate this movie:", max_tokens=200, grammar=grammar)
    ```

=== "Rust"

    ```rust
    use mullama::structured::JsonSchemaConverter;

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "score": { "type": "number", "minimum": 0, "maximum": 100 }
        },
        "required": ["title", "score"]
    });

    let grammar = JsonSchemaConverter::convert(&schema)?;
    println!("{}", grammar);

    let response = context.generate_with_grammar("Rate this movie:", 200, &grammar)?;
    ```

=== "CLI"

    ```bash
    # Convert schema to grammar file, then use it
    mullama grammar from-schema schema.json > output.gbnf
    mullama run llama3.2:1b "Rate this movie:" --grammar output.gbnf
    ```

## Supported Types

The JSON Schema converter supports all standard JSON types:

### Primitive Types

| Type | JSON Schema | Example Output |
|------|-------------|----------------|
| String | `{"type": "string"}` | `"hello world"` |
| Number | `{"type": "number"}` | `3.14` |
| Integer | `{"type": "integer"}` | `42` |
| Boolean | `{"type": "boolean"}` | `true` |
| Null | `{"type": "null"}` | `null` |

### Objects

=== "Node.js"

    ```javascript
    const schema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        address: {
          type: 'object',
          properties: {
            street: { type: 'string' },
            city: { type: 'string' },
            zip: { type: 'string' },
          },
          required: ['street', 'city'],
        },
      },
      required: ['name', 'address'],
    };
    ```

=== "Python"

    ```python
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["street", "city"],
            },
        },
        "required": ["name", "address"],
    }
    ```

=== "Rust"

    ```rust
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "address": {
                "type": "object",
                "properties": {
                    "street": { "type": "string" },
                    "city": { "type": "string" },
                    "zip": { "type": "string" }
                },
                "required": ["street", "city"]
            }
        },
        "required": ["name", "address"]
    });
    ```

=== "CLI"

    ```bash
    # Schema can be passed as a file
    mullama run llama3.2:1b "Extract address:" --schema-file address_schema.json
    ```

### Arrays

=== "Node.js"

    ```javascript
    const schema = {
      type: 'object',
      properties: {
        tags: {
          type: 'array',
          items: { type: 'string' },
        },
        scores: {
          type: 'array',
          items: { type: 'number' },
          minItems: 1,
          maxItems: 5,
        },
      },
    };
    ```

=== "Python"

    ```python
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scores": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
                "maxItems": 5,
            },
        },
    }
    ```

=== "Rust"

    ```rust
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": { "type": "string" }
            },
            "scores": {
                "type": "array",
                "items": { "type": "number" },
                "minItems": 1,
                "maxItems": 5
            }
        }
    });
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "List programming languages:" \
      --schema '{"type":"object","properties":{"languages":{"type":"array","items":{"type":"string"}}}}'
    ```

### Enum Constraints

Restrict string values to a fixed set:

=== "Node.js"

    ```javascript
    const schema = {
      type: 'object',
      properties: {
        sentiment: {
          type: 'string',
          enum: ['positive', 'negative', 'neutral'],
        },
        confidence: { type: 'number' },
      },
      required: ['sentiment', 'confidence'],
    };

    const response = await context.generate(
      "Analyze sentiment: I love this product!",
      100,
      { schema }
    );
    // Output: {"sentiment": "positive", "confidence": 0.95}
    ```

=== "Python"

    ```python
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
    }

    response = context.generate(
        "Analyze sentiment: I love this product!",
        max_tokens=100,
        schema=schema,
    )
    # Output: {"sentiment": "positive", "confidence": 0.95}
    ```

=== "Rust"

    ```rust
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": { "type": "number" }
        },
        "required": ["sentiment", "confidence"]
    });
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Analyze sentiment: I love this product!" \
      --schema '{"type":"object","properties":{"sentiment":{"type":"string","enum":["positive","negative","neutral"]},"confidence":{"type":"number"}},"required":["sentiment","confidence"]}'
    ```

## Required vs Optional Properties

Properties listed in `required` must appear in the output. Optional properties may or may not be included:

=== "Node.js"

    ```javascript
    const schema = {
      type: 'object',
      properties: {
        title: { type: 'string' },         // Required
        author: { type: 'string' },        // Required
        year: { type: 'integer' },         // Optional
        isbn: { type: 'string' },          // Optional
      },
      required: ['title', 'author'],       // Only these are mandatory
    };
    ```

=== "Python"

    ```python
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},       # Required
            "author": {"type": "string"},      # Required
            "year": {"type": "integer"},       # Optional
            "isbn": {"type": "string"},        # Optional
        },
        "required": ["title", "author"],       # Only these are mandatory
    }
    ```

=== "Rust"

    ```rust
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "author": { "type": "string" },
            "year": { "type": "integer" },
            "isbn": { "type": "string" }
        },
        "required": ["title", "author"]
    });
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Extract book info:" --schema-file book.json
    ```

## Integration with Application Code

Parse the structured output and use it in your application:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const schema = {
      type: 'object',
      properties: {
        entities: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              type: { type: 'string', enum: ['person', 'org', 'location'] },
            },
            required: ['name', 'type'],
          },
        },
      },
      required: ['entities'],
    };

    const response = await context.generate(
      "Extract entities: Apple CEO Tim Cook visited Paris last week.",
      300,
      { schema }
    );

    const result = JSON.parse(response);
    for (const entity of result.entities) {
      console.log(`${entity.name} (${entity.type})`);
    }
    // Tim Cook (person)
    // Apple (org)
    // Paris (location)
    ```

=== "Python"

    ```python
    import json
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class Entity:
        name: str
        type: str

    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["person", "org", "location"]},
                    },
                    "required": ["name", "type"],
                },
            },
        },
        "required": ["entities"],
    }

    response = context.generate(
        "Extract entities: Apple CEO Tim Cook visited Paris last week.",
        max_tokens=300,
        schema=schema,
    )

    result = json.loads(response)
    entities = [Entity(**e) for e in result["entities"]]
    for entity in entities:
        print(f"{entity.name} ({entity.type})")
    ```

=== "Rust"

    ```rust
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct ExtractionResult {
        entities: Vec<Entity>,
    }

    #[derive(Deserialize)]
    struct Entity {
        name: String,
        #[serde(rename = "type")]
        entity_type: String,
    }

    let response = context.generate_with_grammar(
        "Extract entities: Apple CEO Tim Cook visited Paris last week.",
        300,
        &grammar
    )?;

    let result: ExtractionResult = serde_json::from_str(&response)?;
    for entity in &result.entities {
        println!("{} ({})", entity.name, entity.entity_type);
    }
    ```

=== "CLI"

    ```bash
    # Pipe output to jq for processing
    mullama run llama3.2:1b \
      "Extract entities: Apple CEO Tim Cook visited Paris last week." \
      --schema-file entities.json | jq '.entities[].name'
    ```

## Error Handling

Grammar-constrained generation always produces valid JSON matching the schema. However, you should still handle potential issues:

=== "Node.js"

    ```javascript
    try {
      const response = await context.generate(prompt, 200, { schema });
      const data = JSON.parse(response);

      // Validate business logic
      if (data.age < 0 || data.age > 150) {
        console.warn('Implausible age value');
      }
    } catch (error) {
      if (error instanceof SyntaxError) {
        // Should not happen with grammar constraints, but be defensive
        console.error('Invalid JSON output');
      } else {
        console.error(`Generation error: ${error.message}`);
      }
    }
    ```

=== "Python"

    ```python
    import json

    try:
        response = context.generate(prompt, max_tokens=200, schema=schema)
        data = json.loads(response)

        # Validate business logic
        if data.get("age", 0) < 0 or data.get("age", 0) > 150:
            print("Warning: implausible age value")
    except json.JSONDecodeError:
        # Should not happen with grammar constraints
        print("Invalid JSON output")
    except Exception as e:
        print(f"Generation error: {e}")
    ```

=== "Rust"

    ```rust
    match context.generate_with_grammar(prompt, 200, &grammar) {
        Ok(response) => {
            match serde_json::from_str::<Value>(&response) {
                Ok(data) => println!("Parsed: {:?}", data),
                Err(e) => eprintln!("Parse error: {}", e),
            }
        }
        Err(e) => eprintln!("Generation error: {}", e),
    }
    ```

=== "CLI"

    ```bash
    # Validate output with jq
    mullama run llama3.2:1b "Extract info:" --schema-file schema.json | jq '.' || echo "Invalid output"
    ```

!!! warning "Max Tokens"
    If `max_tokens` is too low, the output may be truncated before the JSON is complete. Set `max_tokens` high enough to accommodate your schema's maximum possible output size.

## Examples

### Generating API Responses

=== "Node.js"

    ```javascript
    const apiResponseSchema = {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['success', 'error'] },
        data: {
          type: 'object',
          properties: {
            summary: { type: 'string' },
            keywords: { type: 'array', items: { type: 'string' } },
            wordCount: { type: 'integer' },
          },
          required: ['summary', 'keywords'],
        },
      },
      required: ['status', 'data'],
    };

    const response = await context.generate(
      "Summarize: Rust is a systems programming language focused on safety...",
      500,
      { schema: apiResponseSchema }
    );
    ```

=== "Python"

    ```python
    api_response_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "data": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "word_count": {"type": "integer"},
                },
                "required": ["summary", "keywords"],
            },
        },
        "required": ["status", "data"],
    }

    response = context.generate(
        "Summarize: Rust is a systems programming language focused on safety...",
        max_tokens=500,
        schema=api_response_schema,
    )
    ```

=== "Rust"

    ```rust
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "status": { "type": "string", "enum": ["success", "error"] },
            "data": {
                "type": "object",
                "properties": {
                    "summary": { "type": "string" },
                    "keywords": { "type": "array", "items": { "type": "string" } },
                    "word_count": { "type": "integer" }
                },
                "required": ["summary", "keywords"]
            }
        },
        "required": ["status", "data"]
    });
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Summarize: Rust is a systems language..." \
      --schema-file api_response.json
    ```

## See Also

- [Grammar Constraints](grammar.md) -- Write custom GBNF grammars for any format
- [Text Generation](generation.md) -- Core generation parameters
- [Sampling Strategies](sampling.md) -- Combine schemas with sampling control
- [Tutorials: API Server](../examples/api-server.md) -- Using structured output in production
