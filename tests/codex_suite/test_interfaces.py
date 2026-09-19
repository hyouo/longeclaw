import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from longevityclaw.codex_suite.io import SuiteError
from longevityclaw.codex_suite.mcp_server import Server
ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/longeclaw_codex.py"
INIT = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}
READY = {"jsonrpc":"2.0","method":"notifications/initialized"}

def test_cli_valid_json(repo):
    p = subprocess.run([sys.executable,str(RUNNER),"--repo",str(repo),"--workspace",str(repo),"call","list_clocks","--args",'{"modality":"transcriptomics"}'], capture_output=True,text=True,timeout=15)
    assert p.returncode == 0 and json.loads(p.stdout)["data"]["total"] == 3 and p.stderr == ""

def test_cli_error_json(repo):
    p = subprocess.run([sys.executable,str(RUNNER),"--repo",str(repo),"call","list_clocks","--args",'{"invalid":1}'],capture_output=True,text=True,timeout=15)
    assert p.returncode == 2 and not json.loads(p.stdout)["ok"]

def test_cli_bulk(repo):
    p = subprocess.run([sys.executable,str(RUNNER),"--repo",str(repo),"--workspace",str(repo),"bulk","--input","samples.csv","--layout","samples_by_features","--modality","transcriptomics","--scale","tpm","--clocks","toy_rna","--output","bulk.json","--output-csv","bulk.csv"],capture_output=True,text=True,timeout=15)
    assert p.returncode == 0, p.stdout+p.stderr
    assert json.loads(p.stdout)["summary"]["samples"] == 2 and (repo/"bulk.json").is_file() and (repo/"bulk.csv").is_file()

def test_mcp_subprocess(repo):
    requests = [INIT,READY,{"jsonrpc":"2.0","id":2,"method":"tools/list"},{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_clocks","arguments":{"modality":"transcriptomics"}}}]
    p = subprocess.run([sys.executable,str(RUNNER),"--repo",str(repo),"--workspace",str(repo),"mcp"], input="\n".join(json.dumps(r) for r in requests)+"\n",capture_output=True,text=True,timeout=15)
    assert p.returncode == 0 and p.stderr == ""
    replies = [json.loads(line) for line in p.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1,2,3]
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18" and len(replies[1]["result"]["tools"]) == 10
    assert not replies[2]["result"]["isError"] and replies[2]["result"]["structuredContent"]["data"]["total"] == 3

def test_mcp_state_and_negotiation(suite):
    server = Server(suite)
    assert server.handle({"jsonrpc":"2.0","id":0,"method":"tools/list"})["error"]["code"] == -32002
    assert server.handle({**INIT,"params":{**INIT["params"],"protocolVersion":"2099-01-01"}})["result"]["protocolVersion"] == "2025-06-18"
    server.handle(READY)
    assert "tools" in server.handle({"jsonrpc":"2.0","id":2,"method":"tools/list"})["result"]
    assert server.handle(INIT)["error"]["code"] == -32600

@pytest.mark.parametrize("rpc_request", [[],{}, {"jsonrpc":"1.0","id":1,"method":"ping"},{"jsonrpc":"2.0","id":True,"method":"ping"}])
def test_mcp_invalid(suite, rpc_request):
    assert Server(suite).handle(rpc_request)["error"]["code"] == -32600

def test_notifications_never_write(suite, score_args, repo):
    server = Server(suite); server.handle(INIT); server.handle(READY)
    assert server.handle({"jsonrpc":"2.0","method":"tools/call","params":{"name":"score_file","arguments":{**score_args,"output_path":"bad.json"}}}) is None
    assert not (repo/"bad.json").exists()

def test_mcp_tool_validation(suite):
    server = Server(suite); server.handle(INIT); server.handle(READY)
    assert server.handle({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"clock_details","arguments":{}}})["result"]["isError"]

def test_malformed_json_recovery(suite):
    sink = io.StringIO()
    Server(suite).serve(io.BytesIO(b'{broken\n'+json.dumps(INIT).encode()+b'\n'),sink)
    replies = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700 and replies[1]["id"] == 1

def test_permissions_before_import(suite, monkeypatch):
    def forbidden():
        raise AssertionError("must not import blocked tool")
    monkeypatch.setattr(suite,"upstream",forbidden)
    for name in ["pubmed_search","query_longevity_llm","train_custom_model"]:
        with pytest.raises(SuiteError,match="explicit startup"):
            suite.call("upstream_call",{"tool_name":name,"arguments":{}})

def test_upstream_mocked_bridge(suite, monkeypatch):
    module = SimpleNamespace(get_tool_definitions=lambda:[{"name":"get_hallmark_info","input_schema":{"type":"object","properties":{"hallmark_id":{"type":"string"}},"required":["hallmark_id"]}}],get_tool_handlers=lambda:{"get_hallmark_info":lambda hallmark_id:{"hallmark":hallmark_id}})
    monkeypatch.setattr(suite,"upstream",lambda:module)
    assert suite.call("upstream_call",{"tool_name":"get_hallmark_info","arguments":{"hallmark_id":"test"}})["data"]["result"] == {"hallmark":"test"}
    with pytest.raises(SuiteError,match="unknown field"):
        suite.call("upstream_call",{"tool_name":"get_hallmark_info","arguments":{"hallmark_id":"test","bad":1}})

def test_stdout_not_polluted(suite, monkeypatch, capsys):
    def noisy(name,args):
        print("third-party diagnostic")
        return {"ok":True}
    monkeypatch.setattr(suite,"call",noisy)
    server = Server(suite); server.handle(INIT); server.handle(READY)
    result = server.handle({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"doctor"}})
    captured = capsys.readouterr()
    assert captured.out == "" and "third-party diagnostic" in captured.err and not result["result"]["isError"]

def test_real_upstream_assets_when_available():
    if not (ROOT/"CLOCKSdata/unified_aging_clocks.csv").is_file():
        pytest.skip("Full upstream CSV assets unavailable in this build environment")
    from longevityclaw.codex_suite.catalog import Catalog
    catalog = Catalog(ROOT).load()
    assert catalog.models and catalog.list_clocks()["total"] > 0
