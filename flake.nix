{
  description = "tdoa-experiments";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    { }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      rec {
        devShells.default = pkgs.stdenv.mkDerivation {
          name = "tdoa-experiments";
          buildInputs = with pkgs; [
            python313

            # lint, fmt, type
            python313Packages.pylint
            python313Packages.mypy
            python313Packages.black
            gnumake

            # test
            python313Packages.coverage
            python313Packages.pytest

            # libraries used
            python313Packages.numpy
            python313Packages.scipy
            python313Packages.matplotlib
            python313Packages.cartopy
          ];
        };
      }
    );
}
