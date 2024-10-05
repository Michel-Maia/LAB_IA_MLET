param(
    [Parameter(Mandatory=$true)][string]$acrInstance,
    [Parameter(Mandatory=$true)][string]$imageName,
    [Parameter(Mandatory=$false)][string]$imageVersion
)

# Check if the imageVersion variable is set, if not, set it to "latest"
if (-not $imageVersion) {
    $imageVersion = "latest"
}

# Set the working directory to the root of the repository
Set-Location -Path "C:\Users\ricar\Github\azure-ai-tooling"

# Read the .env file
$envFile = Get-Content "src\.env"
$acrAddress = "${acrInstance}.azurecr.io"

# # Extract the username and password from the .env file
$usernameLine = $envFile | Where-Object { $_ -match "^ACR_USER = " }
$passwordLine = $envFile | Where-Object { $_ -match "^ACR_PASSWORD = " }

# # Check if the username and password lines were found
if ($null -ne $usernameLine -and $null -ne $passwordLine) {
    $username = $usernameLine.Split("= ")[3].Trim().Replace('"', '')
    $password = $passwordLine.Split("= ")[3].Trim().Replace('"', '')

    # Login to Azure Container Registry
    az acr login --name $acrInstance --username ${username} --password ${password}

    # Build the Docker image
    docker build -t ${imageName} --file .docker-images/${imageName}/dockerfile .
    docker login $acrAddress

    # Tag the Docker image
    $tag = "${imageName}:${imageVersion}"
    docker tag $imageName $acrAddress/$imageName

    # Push the Docker image
    docker push $acrAddress/$tag
} else {
    Write-Host "Error: Could not find ACR_USER or ACR_PASSWORD in the .env file."
}
