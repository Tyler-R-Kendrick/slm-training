LAKE ?= lake

.PHONY: all build proofs test clean

all: test

build:
	$(LAKE) build

proofs: build
	@command -v rg >/dev/null 2>&1 || { \
	  echo "proof source audit requires rg" >&2; exit 1; \
	}
	@scan_status=0; \
	rg -n '\b(sorry|sorryAx|admit|axiom|unsafe|native_decide)\b' \
	  --glob '*.lean' --glob '!Test/Proofs.lean' . || scan_status=$$?; \
	if [ "$$scan_status" -eq 0 ]; then \
	  echo "forbidden proof escape found" >&2; exit 1; \
	elif [ "$$scan_status" -ne 1 ]; then \
	  echo "proof source audit failed (rg exit $$scan_status)" >&2; \
	  exit "$$scan_status"; \
	fi
	@audit_output=$$(mktemp); trap 'rm -f "$$audit_output"' EXIT HUP INT TERM; \
	if ! $(LAKE) env lean Test/Proofs.lean >"$$audit_output" 2>&1; then \
	  cat "$$audit_output"; exit 1; \
	fi; \
	cat "$$audit_output"; \
	audit_status=0; rg -n '\bsorryAx\b' "$$audit_output" || audit_status=$$?; \
	if [ "$$audit_status" -eq 0 ]; then \
	  echo "theorem audit found sorryAx" >&2; exit 1; \
	elif [ "$$audit_status" -ne 1 ]; then \
	  echo "theorem output audit failed (rg exit $$audit_status)" >&2; \
	  exit "$$audit_status"; \
	fi

test: proofs
	sh Test/run.sh

clean:
	$(LAKE) clean
