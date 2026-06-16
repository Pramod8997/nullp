from graphify.callflow_html import write_callflow_html
from pathlib import Path

def main():
    output_path = write_callflow_html(
        graphify_out='graphify-out',
        graph='graphify-out/graph.json',
        labels='graphify-out/labels.json',
        report='graphify-out/GRAPH_REPORT.md',
        output='graphify-out/graph.html',
        verbose=True
    )
    print(f'HTML visualization generated at {output_path}')

if __name__ == '__main__':
    main()
