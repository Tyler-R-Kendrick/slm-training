LAKE ?= lake

.PHONY: all build proofs test clean

all: test

build:
	$(LAKE) build

proofs: build
	@if rg -n '\b(sorry|admit|axiom|unsafe)\b' --glob '*.lean' \
	  --glob '!Test/Proofs.lean' .; then \
	  echo "forbidden proof escape found" >&2; exit 1; \
	fi
	$(LAKE) env lean Test/Proofs.lean

test: proofs
	sh Test/run.sh

clean:
	$(LAKE) clean
