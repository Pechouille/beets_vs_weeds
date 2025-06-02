#!/bin/bash
cp hooks/* .git/hooks/
chmod +x .git/hooks/pre-commit .git/hooks/pre-push
