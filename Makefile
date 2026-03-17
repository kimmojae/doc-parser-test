.PHONY: build libre run stop clean

build:
	docker build -t libreoffice-server ./libreoffice

libre:
	@docker stop libreoffice-server 2>/dev/null; docker rm libreoffice-server 2>/dev/null; true
	docker run --rm -d --name libreoffice-server libreoffice-server

run:
	.venv/bin/python app.py

stop:
	docker stop libreoffice-server 2>/dev/null || true

clean: stop
	docker rmi libreoffice-server 2>/dev/null || true
