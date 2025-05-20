.PHONY: build clean uninstall

build:
	python build.py

clean:
	rm -rf build dist *.spec

uninstall:
	./dist/secure-note --uninstall
