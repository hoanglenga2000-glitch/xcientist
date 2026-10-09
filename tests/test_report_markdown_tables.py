from evomind_runtime.report_render import render_markdown

def test_generated_markdown_table_rows_are_contiguous():
    document={'title':'fixture','document_sha256':'a'*64}
    content=[{'title':'Evidence','headers':['Tool','Result'],'rows':[['hpc_verify','passed'],['runtime_health','passed']]}]
    value=render_markdown(document,content,[])
    assert '| Tool | Result |\n| --- | --- |\n| hpc_verify | passed |\n| runtime_health | passed |' in value
    assert '|\n\n|' not in value

def test_pipe_characters_and_negative_values_are_preserved():
    value=render_markdown({'title':'fixture','document_sha256':'b'*64},[
        {'title':'Metrics','headers':['Name','Value'],'rows':[['a|b',-1.25]]}],[])
    assert '| a\\|b | -1.25 |' in value
