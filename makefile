.PHONY: build clean uninstall build-patch build-minor build-major

build:
	python scripts/build.py

build-patch:
	python scripts/build.py --increment-patch

build-minor:
	python scripts/build.py --increment-minor

build-major:
	python scripts/build.py --increment-major

clean:
	rm -rf build dist *.spec

uninstall:
	./dist/secure-note --uninstall
