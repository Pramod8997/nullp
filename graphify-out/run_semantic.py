import os
import sys
import json
from pathlib import Path
from graphify.llm import extract_corpus_parallel

def main():
    # 1. Determine Backend based on env keys
    backend = "gemini"
    api_key = None
    
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        backend = "gemini"
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        print("Using Gemini backend...")
    elif os.environ.get("OPENAI_API_KEY"):
        backend = "openai"
        api_key = os.environ.get("OPENAI_API_KEY")
        print("Using OpenAI backend...")
    elif os.environ.get("DEEPSEEK_API_KEY"):
        backend = "deepseek"
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        print("Using Deepseek backend...")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        backend = "claude"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        print("Using Claude backend...")
    else:
        print("> Tip: set `GEMINI_API_KEY` or `GOOGLE_API_KEY` to use Gemini for semantic extraction (`pip install 'graphifyy[gemini]'`).")
        print("Error: No API key found in environment variables (GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY).")
        print("Please export an API key and try again.")
        sys.exit(1)

    uncached_path = Path("graphify-out/.graphify_uncached.txt")
    if not uncached_path.exists():
        print("Error: graphify-out/.graphify_uncached.txt does not exist. Run extract_ast_and_cache.py first.")
        sys.exit(1)
        
    uncached_files = [Path(line.strip()) for line in uncached_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not uncached_files:
        print("No files need semantic extraction.")
        # Create empty semantic results
        semantic_result = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    else:
        print(f"Starting semantic extraction for {len(uncached_files)} files...")
        semantic_result = extract_corpus_parallel(
            uncached_files,
            backend=backend,
            api_key=api_key,
            root=Path("."),
            token_budget=60000,
            max_concurrency=4
        )
    
    Path("graphify-out/.graphify_semantic.json").write_text(
        json.dumps(semantic_result, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f"Semantic extraction completed: {len(semantic_result.get('nodes', []))} nodes, {len(semantic_result.get('edges', []))} edges.")

    # 2. Merge with AST
    print("Merging AST and Semantic extractions...")
    ast_path = Path("graphify-out/.graphify_ast.json")
    if not ast_path.exists():
        print("Warning: graphify-out/.graphify_ast.json not found. Treating as empty.")
        ast_result = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    else:
        ast_result = json.loads(ast_path.read_text(encoding='utf-8'))

    # Merge nodes
    merged_nodes = {}
    for node in ast_result.get("nodes", []):
        merged_nodes[node["id"]] = node

    for node in semantic_result.get("nodes", []):
        nid = node["id"]
        if nid in merged_nodes:
            # Merge semantic metadata into AST node
            for k, v in node.items():
                if v is not None and merged_nodes[nid].get(k) is None:
                    merged_nodes[nid][k] = v
        else:
            merged_nodes[nid] = node

    # Merge edges
    merged_edges = {}
    for edge in ast_result.get("edges", []):
        key = (edge["source"], edge["target"])
        merged_edges[key] = edge

    for edge in semantic_result.get("edges", []):
        key = (edge["source"], edge["target"])
        if key not in merged_edges:
            merged_edges[key] = edge

    # Merge hyperedges
    merged_hyperedges = ast_result.get("hyperedges", []) + semantic_result.get("hyperedges", [])

    merged_result = {
        "nodes": list(merged_nodes.values()),
        "edges": [
            {
                "source": k[0],
                "target": k[1],
                "relation": v.get("relation", "calls"),
                "confidence": v.get("confidence", "EXTRACTED"),
                "confidence_score": v.get("confidence_score", 1.0),
                "source_file": v.get("source_file"),
                "source_location": v.get("source_location"),
                "weight": v.get("weight", 1.0)
            }
            for k, v in merged_edges.items()
        ],
        "hyperedges": merged_hyperedges,
        "input_tokens": ast_result.get("input_tokens", 0) + semantic_result.get("input_tokens", 0),
        "output_tokens": ast_result.get("output_tokens", 0) + semantic_result.get("output_tokens", 0)
    }

    Path("graphify-out/.graphify_extract.json").write_text(
        json.dumps(merged_result, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f"Merged result saved to graphify-out/.graphify_extract.json: {len(merged_result['nodes'])} nodes, {len(merged_result['edges'])} edges.")

if __name__ == '__main__':
    main()
