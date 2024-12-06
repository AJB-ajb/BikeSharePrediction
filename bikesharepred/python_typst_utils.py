import numpy as np

def typst_str(ob, in_code = False, float_format = '${num:.2e}$'):
    if in_code:
        return f"[{typst_str(ob, in_code = False)}]"
    if type(ob) == str:
        return ob
    # if ob is any number type, enclose in $ $:
    elif type(ob) == int:
        return f"${ob}$"
    elif type(ob) == float or np.issubdtype(type(ob), np.number):
        return float_format.format(num = ob)
    else:
        return str(ob)
    
"""
Good Practises for typst tables of programmatic output:
- use #show table: set text(size: …) to set the text to a good size, such that the table is rendered consistently
- use ` ` to enclose text inside for generated output
"""
def print_typst_table(matrix, col_sizings = None, row_name = None, col_names = None):
    """
        matrix: numpy array of numbers | strings | general object accepted by the typst_str function
        row_name_func: row index -> typst formattable object
        col_names: list of column names
    """
    matrix = np.array(matrix)
    np_typst_str = np.frompyfunc(typst_str, 1, 1)
    matrix = np_typst_str(matrix)
    if not row_name is None:
        row_names = np.array([typst_str(row_name(i)) for i in range(matrix.shape[0])])
        matrix = np.concatenate((np.reshape(row_names, (len(row_names), 1)), matrix), axis = 1) 
    if col_sizings is None:
        cols = str(matrix.shape[1]) 
    else:
        cols = "(" + ", ".join(col_sizings for _ in range(matrix.shape[1])) + ")"

    header = ""
    if col_names is not None:
        header_str = ", ".join(typst_str(col, in_code=True) for col in col_names)
        header = f"\ntable.header({header_str}),"
        
    format_row = lambda vals: ", ".join(typst_str(val, in_code = True) for val in vals)
    print(f"#table(columns: {cols},{header}\n", ",\n".join(format_row(matrix[i, :]) for i in range(matrix.shape[0])),")", sep = '')