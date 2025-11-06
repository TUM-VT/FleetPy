import requests
import zipfile
from pathlib import Path
import shutil


def check_manhattan_data(fleetpy_path):
    """Check if Manhattan data exists and download if missing"""
    # Define required directories
    data_dir = Path(fleetpy_path) / 'data'
    required_paths = {
        'demand': data_dir / 'demand' / 'Manhattan_2018',
        'network': data_dir / 'networks' / 'Manhattan_2019_corrected',
        'zones': [
            data_dir / 'zones' / 'Manhattan_corrected_6min_max',
            data_dir / 'zones' / 'Manhattan_corrected_8min_min',
            data_dir / 'zones' / 'Manhattan_corrected_12min_max',
            data_dir / 'zones' / 'Manhattan_Taxi_Zones'
        ]
    }
    
    # Check if any required data is missing
    missing_data = False
    for category, path in required_paths.items():
        if isinstance(path, list):
            if not any(p.exists() for p in path):
                print(f"❌ Missing {category} data")
                missing_data = True
        elif not path.exists():
            print(f"❌ Missing {category} data")
            missing_data = True
    
    if not missing_data:
        print("✅ All Manhattan data found!")
        return True
    
    # Download the data
    print("\nDownloading Manhattan benchmark dataset...")
    zenodo_url = "https://zenodo.org/records/15187906/files/FleetPy_Manhattan.zip?download=1"
    
    try:
        # Create data directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the zip file
        temp_zip = data_dir / 'manhattan_temp.zip'
        response = requests.get(zenodo_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                # Print progress
                if total_size > 0:
                    progress = int(50 * downloaded / total_size)
                    print(f"\rDownloading: [{'=' * progress}{' ' * (50-progress)}] {downloaded/total_size*100:.1f}%", end='')
        
        print("\nExtracting files...")
        # Create a temporary directory for extraction
        temp_dir = data_dir / 'temp_extract'
        temp_dir.mkdir(exist_ok=True)
        
        # Extract the zip file to temporary directory
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Move contents to their final locations
        fleetpy_manhattan = temp_dir / 'FleetPy_Manhattan'
        if fleetpy_manhattan.exists():
            # Move demand files
            demand_src = fleetpy_manhattan / 'demand'
            if demand_src.exists():
                print("Moving demand data...")
                shutil.copytree(demand_src, data_dir / 'demand', dirs_exist_ok=True)
            
            # Move network files
            network_src = fleetpy_manhattan / 'networks'
            if network_src.exists():
                print("Moving network data...")
                shutil.copytree(network_src, data_dir / 'networks', dirs_exist_ok=True)
            
            # Move zones files
            zones_src = fleetpy_manhattan / 'zones'
            if zones_src.exists():
                print("Moving zones data...")
                shutil.copytree(zones_src, data_dir / 'zones', dirs_exist_ok=True)
        
        # Clean up
        temp_zip.unlink()
        shutil.rmtree(temp_dir)
        
        print("✅ Successfully downloaded and extracted Manhattan data to appropriate directories!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {str(e)}")
        print("Please download manually from: https://zenodo.org/records/15187906/files/FleetPy_Manhattan.zip?download=1")
        print("and extract to the FleetPy/data/ directory")
        return False
