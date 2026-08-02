LAKE ?= lake
RG ?= rg
GREP ?= grep

.PHONY: all build proofs test clean

all: test

build:
	$(LAKE) build

proofs: build
	@scan_status=0; scan_tool=; \
	if command -v $(RG) >/dev/null 2>&1; then \
	  scan_tool=rg; \
	  $(RG) -n '\b(sorry|sorryAx|admit|axiom|unsafe|native_decide)\b' \
	    --glob '*.lean' --glob '!Test/Proofs.lean' . || scan_status=$$?; \
	elif command -v $(GREP) >/dev/null 2>&1; then \
	  scan_tool=grep; \
	  $(GREP) -REn --include='*.lean' --exclude='Proofs.lean' \
	    '(^|[^[:alnum:]_])(sorry|sorryAx|admit|axiom|unsafe|native_decide)([^[:alnum:]_]|$$)' \
	    . || scan_status=$$?; \
	else \
	  echo "proof source audit requires rg or grep" >&2; exit 1; \
	fi; \
	if [ "$$scan_status" -eq 0 ]; then \
	  echo "forbidden proof escape found" >&2; exit 1; \
	elif [ "$$scan_status" -ne 1 ]; then \
	  echo "proof source audit failed ($$scan_tool exit $$scan_status)" >&2; \
	  exit "$$scan_status"; \
	fi
	@audit_output=$$(mktemp); trap 'rm -f "$$audit_output"' EXIT HUP INT TERM; \
	if ! $(LAKE) env lean Test/Proofs.lean >"$$audit_output" 2>&1; then \
	  cat "$$audit_output"; exit 1; \
	fi; \
	cat "$$audit_output"; \
	audit_status=0; audit_tool=; \
	if command -v $(RG) >/dev/null 2>&1; then \
	  audit_tool=rg; $(RG) -n '\bsorryAx\b' "$$audit_output" || audit_status=$$?; \
	elif command -v $(GREP) >/dev/null 2>&1; then \
	  audit_tool=grep; $(GREP) -En '(^|[^[:alnum:]_])sorryAx([^[:alnum:]_]|$$)' \
	    "$$audit_output" || audit_status=$$?; \
	else \
	  echo "theorem output audit requires rg or grep" >&2; exit 1; \
	fi; \
	if [ "$$audit_status" -eq 0 ]; then \
	  echo "theorem audit found sorryAx" >&2; exit 1; \
	elif [ "$$audit_status" -ne 1 ]; then \
	  echo "theorem output audit failed ($$audit_tool exit $$audit_status)" >&2; \
	  exit "$$audit_status"; \
	fi

test: proofs
	sh Test/run.sh

clean:
	$(LAKE) clean
