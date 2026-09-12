.PHONY: all build build-release install uninstall bump build-linux test test-install \
	test-abi check flatpak flatpak-lint fix shaders screenshot clean size

APP_NAME := bnkterm
APP_ID := io.github.borgenk.BnkTerm
TARGET := x86_64-unknown-linux-gnu
VERSION := $(shell grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
BUILD_PATH := target/$(TARGET)/release
TARBALL := $(APP_NAME)-v$(VERSION)-$(TARGET).tar.gz
BIN_DIR := ~/.local/bin
APPS_DIR := ~/.local/share/applications
ICON_DIR := ~/.local/share/icons/hicolor/scalable/apps

# The oldest glibc a downloaded tarball runs on, which is Debian 12's. The
# release builds in that image and checks the binary against this.
GLIBC_FLOOR := 2.36

FLATPAK_BUILDER := $(shell command -v flatpak-builder 2>/dev/null || echo "flatpak run org.flatpak.Builder")
FLATPAK_BUILDER_LINT := $(shell command -v flatpak-builder-lint 2>/dev/null \
	|| echo "flatpak run --command=flatpak-builder-lint org.flatpak.Builder")
LINT_EXCEPTIONS := --exceptions --user-exceptions .github/flatpak-lint-exceptions.json
RUNTIME_REPO := https://flathub.org/repo/flathub.flatpakrepo

all: test build

build:
	cargo build

build-release:
	cargo build --release

# Install the release binary, desktop entry and icon under ~/.local, the same
# per-user location install.sh uses.
install: build-release
	install -Dm755 $(BUILD_PATH)/$(APP_NAME) $(BIN_DIR)/$(APP_NAME)
	install -Dm644 assets/$(APP_NAME).desktop $(APPS_DIR)/$(APP_NAME).desktop
	install -Dm644 assets/$(APP_NAME).svg $(ICON_DIR)/$(APP_NAME).svg
	-update-desktop-database $(APPS_DIR)
	@echo "Installed $(APP_NAME) to $(BIN_DIR)/"
	@echo "Installed desktop file + icon to ~/.local/share/"
	@if command -v fc-list > /dev/null 2>&1 && ! fc-list | grep -qi 'symbols nerd font'; then \
		echo "Note: no Symbols Nerd Font found, so the password-lock cursor is off."; \
	fi

uninstall:
	rm -f $(BIN_DIR)/$(APP_NAME) $(APPS_DIR)/$(APP_NAME).desktop $(ICON_DIR)/$(APP_NAME).svg
	-update-desktop-database $(APPS_DIR)
	@echo "Removed $(APP_NAME) from $(BIN_DIR)/ and ~/.local/share/"

# Bump the version, commit, and tag: make bump V=0.2.0
# The release workflow rejects a tag that does not match this version, and the
# metainfo entry is the release history the Flatpak reports.
bump:
	@test -n "$(V)" || (echo "Current: $(VERSION). Usage: make bump V=0.2.0" && exit 1)
	sed -i '0,/^version = ".*"/{s//version = "$(V)"/}' Cargo.toml
	sed -i 's|<releases>|<releases>\n    <release version="$(V)" date="'"$$(date -u +%F)"'"/>|' assets/$(APP_ID).metainfo.xml
	cargo update --workspace
	git add Cargo.toml Cargo.lock assets/$(APP_ID).metainfo.xml
	git commit -m "Bump version to $(V)"
	git tag "v$(V)"
	@echo "Bumped to v$(V). Push with: git push origin main --tags"

# Build a release tarball into dist/ for upload to a GitHub Release, with the
# checksum install.sh verifies beside it. The layout is install.sh's contract.
build-linux: build-release
	sh .github/scripts/check-abi.sh $(GLIBC_FLOOR) $(BUILD_PATH)/$(APP_NAME)
	rm -rf dist/stage
	mkdir -p dist/stage/icons/hicolor/scalable/apps
	cp $(BUILD_PATH)/$(APP_NAME) dist/stage/$(APP_NAME)
	cp assets/$(APP_NAME).desktop dist/stage/$(APP_NAME).desktop
	cp assets/$(APP_NAME).svg dist/stage/icons/hicolor/scalable/apps/$(APP_NAME).svg
	tar czf dist/$(TARBALL) -C dist/stage .
	rm -rf dist/stage
	cd dist && sha256sum $(TARBALL) > $(TARBALL).sha256
	@echo "Built dist/$(TARBALL), with a .sha256"
	@echo "Publish with: gh release create v$(VERSION) dist/$(TARBALL)*"

# The bundle the release workflow publishes, into dist/.
flatpak:
	mkdir -p dist
	$(FLATPAK_BUILDER) --force-clean --repo=flatpak-repo build-dir $(APP_ID).yml
	flatpak build-bundle --runtime-repo=$(RUNTIME_REPO) flatpak-repo \
		dist/$(APP_NAME)-v$(VERSION)-x86_64.flatpak $(APP_ID)
	@echo "Built dist/$(APP_NAME)-v$(VERSION)-x86_64.flatpak"
	@echo "Install with: flatpak install --user dist/$(APP_NAME)-v$(VERSION)-x86_64.flatpak"

# The manifest and the built repo, against Flathub's rules.
flatpak-lint: flatpak
	$(FLATPAK_BUILDER_LINT) $(LINT_EXCEPTIONS) manifest $(APP_ID).yml
	$(FLATPAK_BUILDER_LINT) $(LINT_EXCEPTIONS) repo flatpak-repo

# The gate. rustdoc is in it because it is what checks the intra-doc links.
test:
	cargo fmt --check
	cargo clippy --all --benches --tests --examples --all-features -- -D clippy::all -D warnings
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --all-features
	cargo test

# The installer, over file:// URLs, with no network.
test-install:
	sh .github/scripts/test-install.sh

# The ABI floor check, against binaries this machine already has.
test-abi:
	sh .github/scripts/test-abi.sh

check: test test-install test-abi

fix:
	cargo fmt
	cargo clippy --all --benches --tests --examples --all-features --fix --allow-dirty

screenshot:
	cargo test -- --ignored --nocapture write_screenshot

# Recompile the committed SPIR-V. Needs glslc, and only when a shader changes.
shaders:
	glslc -O shaders/quad.vert -o shaders/quad.vert.spv
	glslc -O shaders/quad.frag -o shaders/quad.frag.spv

clean:
	cargo clean

size:
	@ls -lh $(BUILD_PATH)/$(APP_NAME) 2>/dev/null || echo "Not built"
