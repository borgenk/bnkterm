.PHONY: all build build-release install uninstall build-linux test test-install \
	test-abi check fix shaders clean size

APP_NAME := bnkterm
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

# Recompile the committed SPIR-V. Needs glslc, and only when a shader changes.
shaders:
	glslc -O shaders/quad.vert -o shaders/quad.vert.spv
	glslc -O shaders/quad.frag -o shaders/quad.frag.spv

clean:
	cargo clean

size:
	@ls -lh $(BUILD_PATH)/$(APP_NAME) 2>/dev/null || echo "Not built"
