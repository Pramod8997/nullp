import os
import sys
import json
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate

def get_fallback_labels(communities, G):
    labels = {}
    for cid, nodes in communities.items():
        # Heuristic: find the node with the highest degree in the community
        best_node = None
        best_deg = -1
        for node in nodes:
            if node in G:
                deg = G.degree(node)
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
        
        if best_node:
            # Clean up the node ID for a label
            label = G.nodes[best_node].get('label', best_node)
            labels[cid] = f"Group: {label}"
        else:
            labels[cid] = f"Community {cid}"
    return labels

def main():
    extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text(encoding="utf-8"))
    detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding="utf-8"))
    analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text(encoding="utf-8"))

    G = build_from_json(extraction)
    communities = {int(k): v for k, v in analysis['communities'].items()}
    cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
    tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

    # Try to label using Gemini
    labels = {}
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("Calling Gemini to label communities...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            
            # Prepare data for Gemini
            prompt_data = []
            for cid, nodes in communities.items():
                node_details = []
                for n in nodes[:10]: # Cap at 10 nodes per community to avoid token limits
                    if n in G:
                        label = G.nodes[n].get('label', n)
                        source = G.nodes[n].get('source_file', '')
                        node_details.append(f"{label} (in {source})")
                prompt_data.append({
                    "id": cid,
                    "nodes": node_details
                })
            
            prompt = (
                "You are an expert software architect. Below is a list of code/doc communities (clusters of related functions, classes, and documentation) detected in a codebase.\n"
                "For each community, provide a concise, professional 2-5 word label that summarizes its technical domain or architectural purpose (e.g. 'Authentication & Session Handling', 'Database Session Manager', 'CT Sensor Calibration').\n"
                "Return ONLY a JSON object mapping the community ID (as string) to its name. Do not include markdown formatting or explanation.\n\n"
                f"{json.dumps(prompt_data, indent=2)}"
            )
            
            resp = client.chat.completions.create(
                model="gemini-3-flash-preview",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
            
            gemini_labels = json.loads(content)
            for cid_str, name in gemini_labels.items():
                labels[int(cid_str)] = name
            print(f"Successfully labeled {len(labels)} communities using Gemini.")
        except Exception as e:
            print(f"Error during Gemini labeling: {e}. Falling back to heuristics...")
            labels = get_fallback_labels(communities, G)
    else:
        print("No API key found. Using fallback heuristics for labeling...")
        labels = get_fallback_labels(communities, G)

    # Ensure all communities are labeled
    for cid in communities:
        if cid not in labels:
            labels[cid] = f"Community {cid}"

    # Print labels summary
    print("\nCommunity Labels:")
    for cid, label in sorted(labels.items())[:15]:
        print(f"  Community {cid}: {label} ({len(communities[cid])} nodes)")
    if len(labels) > 15:
        print(f"  ... and {len(labels) - 15} more.")

    # Regenerate questions and report
    questions = suggest_questions(G, communities, labels)
    report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, '.', suggested_questions=questions)
    
    Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding="utf-8")
    Path('graphify-out/labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\nRegenerated GRAPH_REPORT.md and saved graphify-out/labels.json.")

if __name__ == '__main__':
    main()
