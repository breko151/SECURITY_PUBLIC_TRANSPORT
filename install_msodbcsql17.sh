#!/bin/bash

# Añadir el repositorio de Microsoft
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Actualizar e instalar msodbcsql17
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17