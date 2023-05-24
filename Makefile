.PHONY: build
build:
	docker-compose -f ./docker-compose.yml build fairdro

.PHONY: up
up:
	docker-compose -f ./docker-compose.yml up -d fairdro

.PHONY: exec
exec:
	docker exec -it fairdro bash

.PHONY: down
down:
	docker-compose -f ./docker-compose.yml down
