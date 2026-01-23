# Grammar Constraints

Use GBNF (Grammar Backus-Naur Form) grammars to constrain model output to specific formats and structures. This is a powerful mechanism for ensuring the model produces syntactically valid output for any format you define.

## What is GBNF?

GBNF is a grammar format used by llama.cpp to constrain token generation. During generation, the model can only select tokens that are valid according to the grammar, guaranteeing the output matches the defined structure.

## Grammar Syntax

GBNF grammars consist of rules that define the structure of valid output:

```
# Rules are defined as: name ::= expression
root   ::= value
value  ::= string | number | "true" | "false" | "null"
string ::= "\"" [^"\\]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
```

### Syntax Elements

| Element | Description | Example |
|---------|-------------|---------|
| `"text"` | Terminal (literal string) | `"hello"` |
| `[chars]` | Character class | `[a-zA-Z0-9]` |
| `[^chars]` | Negated character class | `[^"\\]` |
| `A B` | Sequence | `"[" value "]"` |
| `A \| B` | Alternation (OR) | `"true" \| "false"` |
| `A*` | Zero or more | `[0-9]*` |
| `A+` | One or more | `[0-9]+` |
| `A?` | Optional (zero or one) | `"-"?` |
| `(A B)` | Grouping | `("," value)*` |

## Using Grammars

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    // Define a grammar for a simple list format
    const grammar = `
    root   ::= item ("\\n" item)*
    item   ::= "- " content
    content ::= [^\\n]+
    `;

    const response = await context.generate(
      "List 5 programming languages:",
      200,
      { grammar }
    );
    console.log(response);
    // - Python
    // - Rust
    // - JavaScript
    // - Go
    // - TypeScript
    ```

=== "Python"

    ```python
    from mullama import Model, Context

    model = Model.load("./model.gguf")
    context = Context(model)

    # Define a grammar for a simple list format
    grammar = r"""
    root   ::= item ("\n" item)*
    item   ::= "- " content
    content ::= [^\n]+
    """

    response = context.generate(
        "List 5 programming languages:",
        max_tokens=200,
        grammar=grammar,
    )
    print(response)
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams};

    let grammar = r#"
    root   ::= item ("\n" item)*
    item   ::= "- " content
    content ::= [^\n]+
    "#;

    let response = context.generate_with_grammar(
        "List 5 programming languages:",
        200,
        grammar
    )?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    # Grammar from a file
    mullama run llama3.2:1b "List 5 programming languages:" \
      --grammar list.gbnf

    # Inline grammar
    mullama run llama3.2:1b "List 5 languages:" \
      --grammar 'root ::= item ("\n" item)* \n item ::= "- " [^\n]+'
    ```

## Common Grammar Patterns

### JSON Object

```
root   ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair   ::= string ws ":" ws value
value  ::= string | number | "true" | "false" | "null" | array | object
string ::= "\"" [^"\\]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
array  ::= "[" ws (value ("," ws value)*)? ws "]"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
ws     ::= [ \t\n]*
```

### CSV Format

```
root   ::= header "\n" rows
header ::= cell ("," cell)*
rows   ::= row ("\n" row)*
row    ::= cell ("," cell)*
cell   ::= [^,\n]*
```

### Email Address

```
root    ::= local "@" domain
local   ::= [a-zA-Z0-9._%+-]+
domain  ::= label ("." label)+
label   ::= [a-zA-Z0-9-]+
```

### ISO Date

```
root  ::= year "-" month "-" day
year  ::= [0-9][0-9][0-9][0-9]
month ::= "0" [1-9] | "1" [0-2]
day   ::= "0" [1-9] | [1-2] [0-9] | "3" [0-1]
```

### Key-Value Pairs

```
root  ::= pair ("\n" pair)*
pair  ::= key ": " value
key   ::= [a-zA-Z_][a-zA-Z0-9_]*
value ::= [^\n]+
```

## Creating Custom Grammars

Build grammars tailored to your application's output format:

=== "Node.js"

    ```javascript
    // Grammar for a structured review
    const reviewGrammar = `
    root     ::= rating "\\n" summary "\\n" pros "\\n" cons
    rating   ::= "Rating: " [1-5] "/5"
    summary  ::= "Summary: " sentence
    pros     ::= "Pros: " items
    cons     ::= "Cons: " items
    items    ::= item (", " item)*
    item     ::= [a-zA-Z ]+
    sentence ::= [A-Z] [a-zA-Z ,.]+ "."
    `;

    const response = await context.generate(
      "Review this restaurant: Great food but slow service.",
      300,
      { grammar: reviewGrammar }
    );
    ```

=== "Python"

    ```python
    # Grammar for a structured review
    review_grammar = r"""
    root     ::= rating "\n" summary "\n" pros "\n" cons
    rating   ::= "Rating: " [1-5] "/5"
    summary  ::= "Summary: " sentence
    pros     ::= "Pros: " items
    cons     ::= "Cons: " items
    items    ::= item (", " item)*
    item     ::= [a-zA-Z ]+
    sentence ::= [A-Z] [a-zA-Z ,.]+ "."
    """

    response = context.generate(
        "Review this restaurant: Great food but slow service.",
        max_tokens=300,
        grammar=review_grammar,
    )
    ```

=== "Rust"

    ```rust
    let review_grammar = r#"
    root     ::= rating "\n" summary "\n" pros "\n" cons
    rating   ::= "Rating: " [1-5] "/5"
    summary  ::= "Summary: " sentence
    pros     ::= "Pros: " items
    cons     ::= "Cons: " items
    items    ::= item (", " item)*
    item     ::= [a-zA-Z ]+
    sentence ::= [A-Z] [a-zA-Z ,.]+ "."
    "#;

    let response = context.generate_with_grammar(
        "Review this restaurant: Great food but slow service.",
        300,
        review_grammar
    )?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b \
      "Review this restaurant: Great food but slow service." \
      --grammar review.gbnf
    ```

## Grammar Validation

Validate grammars before use to catch syntax errors:

=== "Node.js"

    ```javascript
    import { Grammar } from 'mullama';

    try {
      const grammar = Grammar.parse(`
        root ::= "hello" | "world"
      `);
      console.log('Grammar is valid');
    } catch (error) {
      console.error(`Grammar error: ${error.message}`);
    }
    ```

=== "Python"

    ```python
    from mullama import Grammar, GrammarError

    try:
        grammar = Grammar.parse("""
            root ::= "hello" | "world"
        """)
        print("Grammar is valid")
    except GrammarError as e:
        print(f"Grammar error: {e}")
    ```

=== "Rust"

    ```rust
    use mullama::Grammar;

    match Grammar::parse(r#"root ::= "hello" | "world""#) {
        Ok(_) => println!("Grammar is valid"),
        Err(e) => eprintln!("Grammar error: {}", e),
    }
    ```

=== "CLI"

    ```bash
    # Validate a grammar file
    mullama grammar validate my_grammar.gbnf
    ```

## Performance Impact

Grammar constraints add a small overhead per token due to grammar state tracking:

| Scenario | Overhead | Notes |
|----------|----------|-------|
| No grammar | Baseline | Fastest generation |
| Simple grammar | ~5% slower | Few rules, simple patterns |
| Complex grammar | ~10-15% slower | Many rules, deep nesting |
| Very complex | ~20% slower | Extensive character classes, recursion |

!!! tip "Performance Tips"
    - Keep grammars as simple as possible
    - Avoid deeply recursive rules when a flat structure works
    - Use character classes (`[a-z]+`) instead of repeated alternatives (`"a" \| "b" \| ...`)
    - For JSON output, prefer the built-in `--schema` option over manual JSON grammars

## Combining with Sampling Parameters

Grammar constraints work alongside sampling parameters. The grammar filters invalid tokens, then sampling selects among the remaining valid options:

=== "Node.js"

    ```javascript
    const response = await context.generate("Generate data:", 200, {
      grammar: myGrammar,
      temperature: 0.7,   // Controls variety within valid grammar outputs
      topK: 40,           // Further limits candidates among valid tokens
    });
    ```

=== "Python"

    ```python
    response = context.generate("Generate data:", max_tokens=200,
        grammar=my_grammar,
        params=SamplerParams(temperature=0.7, top_k=40))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_grammar_and_params(
        "Generate data:", 200, grammar,
        SamplerParams { temperature: 0.7, top_k: 40, ..Default::default() }
    )?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Generate data:" \
      --grammar my_grammar.gbnf \
      --temperature 0.7 \
      --top-k 40
    ```

## Loading Grammars from Files

Load grammar definitions from `.gbnf` files:

=== "Node.js"

    ```javascript
    import { readFileSync } from 'fs';

    const grammar = readFileSync('./grammars/json.gbnf', 'utf-8');
    const response = await context.generate("Output JSON:", 200, { grammar });
    ```

=== "Python"

    ```python
    from pathlib import Path

    grammar = Path("./grammars/json.gbnf").read_text()
    response = context.generate("Output JSON:", max_tokens=200, grammar=grammar)
    ```

=== "Rust"

    ```rust
    let grammar = std::fs::read_to_string("grammars/json.gbnf")?;
    let response = context.generate_with_grammar("Output JSON:", 200, &grammar)?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Output JSON:" --grammar grammars/json.gbnf
    ```

## Examples

### Regex-Like Pattern: Phone Number

```
root  ::= "(" area ") " prefix "-" line
area  ::= [0-9][0-9][0-9]
prefix ::= [0-9][0-9][0-9]
line  ::= [0-9][0-9][0-9][0-9]
```

### Structured Data: SQL Query

```
root    ::= select from where? orderby? ";"
select  ::= "SELECT " columns
from    ::= " FROM " table
where   ::= " WHERE " condition
orderby ::= " ORDER BY " column " " direction
columns ::= column (", " column)*
column  ::= [a-z_]+
table   ::= [a-z_]+
condition ::= column " " operator " " value
operator ::= "=" | "!=" | ">" | "<" | ">=" | "<="
value   ::= "'" [^']* "'" | [0-9]+
direction ::= "ASC" | "DESC"
```

### Markdown List with Categories

```
root     ::= category+
category ::= "## " title "\n" items "\n"
title    ::= [A-Z][a-zA-Z ]+
items    ::= item ("\n" item)*
item     ::= "- " description
description ::= [^\n]+
```

## See Also

- [Structured Output](structured-output.md) -- JSON Schema-based output (uses grammars internally)
- [Sampling Strategies](sampling.md) -- Combining grammars with sampling
- [Text Generation](generation.md) -- Core generation parameters
- [API Reference: Configuration](../api/configuration.md) -- Grammar configuration options
