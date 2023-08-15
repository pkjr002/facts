import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import Preprocessor
import copy

class RemoveCodeCellPreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == 'code':
            cell = copy.deepcopy(cell)
            cell.source = ''
            cell.metadata = {}
        return cell, resources

# Custom CSS to remove cell outlines
custom_css = """
<style>
    .cell, .input_area, .output_area, .output, .prompt, .input, .output_wrapper {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
</style>
"""

# Load the notebook
with open('NZ_IP_location.ipynb', 'r', encoding='utf-8') as file:
    notebook_content = nbformat.read(file, as_version=4)

# Create the custom HTML exporter
html_exporter = HTMLExporter()
html_exporter.register_preprocessor(RemoveCodeCellPreprocessor, enabled=True)

# Convert to HTML
(html_body, _) = html_exporter.from_notebook_node(notebook_content)

# Add custom CSS to the HTML head
html_body = html_body.replace('</head>', custom_css + '</head>')

# Save the HTML
with open('output.html', 'w', encoding='utf-8') as file:
    file.write(html_body)
