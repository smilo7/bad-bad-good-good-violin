import nbformat

notebook_path = "Violin_Quality_Classifier_with_Augmentations.ipynb"
nb = nbformat.read(notebook_path, as_version=4)
new_cells = []

for cell in nb.cells:
    if cell.cell_type == "code" and cell.source.startswith("#@title"):
        lines = cell.source.splitlines()
        title_line = lines[0]
        title_text = title_line.replace("#@title", "").strip()
        # Remove the title line from code
        code_body = "\n".join(lines[1:]).lstrip()
        # Insert markdown cell above
        new_cells.append(nbformat.v4.new_markdown_cell(f"### {title_text}"))
        # Insert code cell without title
        cell.source = code_body
        new_cells.append(cell)
    else:
        new_cells.append(cell)

nb.cells = new_cells
nbformat.write(nb, notebook_path)
print("Notebook updated: #@title lines converted to markdown headers.")