#!/bin/bash
#
# Master script to fetch files and submit Condor jobs
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

echo "=================================================="
echo "CMS Track Purity - Condor Job Submission"
echo "=================================================="
echo ""

# Step 1: Fetch file lists from DAS
echo "Step 1: Fetching file lists from DAS..."
echo "This may take several minutes..."
echo ""

if [ ! -f "fetch_dataset_files.py" ]; then
    echo "ERROR: fetch_dataset_files.py not found!"
    exit 1
fi

# python3 fetch_dataset_files.py

if [ ! -f "file_lists/all_jobs.txt" ]; then
    echo "ERROR: Job list file not created!"
    exit 1
fi

# Count total jobs
TOTAL_JOBS=$(wc -l < file_lists/all_jobs.txt)
echo ""
echo "Total jobs to submit: $TOTAL_JOBS"
echo ""

# Step 2: Create output directories
echo "Step 2: Creating output directories..."
mkdir -p logs
mkdir -p /ceph/cms/store/user/aaarora/tracking

# Make scripts executable
chmod +x run_cmsdriver_job.sh
chmod +x fetch_dataset_files.py

# Step 3: Submit jobs
echo ""
echo "Step 3: Submitting jobs to Condor..."
echo ""

read -p "Do you want to submit $TOTAL_JOBS jobs? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Submission cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
condor_submit condor_submit.sub

echo ""
echo "=================================================="
echo "Jobs submitted successfully!"
echo ""
echo "Monitor jobs with:"
echo "  condor_q"
echo ""
echo "Check specific user jobs:"
echo "  condor_q -submitter \$USER"
echo ""
echo "View job details:"
echo "  condor_q -analyze <job_id>"
echo ""
echo "Remove all jobs:"
echo "  condor_rm \$USER"
echo ""
echo "Output files will be in:"
echo "  /ceph/cms/store/user/aaarora/tracking/<dataset_name>/"
echo ""
echo "Logs will be in:"
echo "  $SCRIPT_DIR/logs/"
echo "=================================================="
