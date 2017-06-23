package main.result.batch;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 * One ExperimentResult represents results for 1 attribute subset
**/

public abstract class ExperimentResult {
    protected double mean;
    protected double sd;
    protected double maxDist;
    protected Instance maxInstance;
    protected double minDist;
    protected Instance minInstance;
    protected double distance;

    protected double[] separateDistance;
    protected Instances instances;

    ExperimentResult(double distance, double[] separateDistance, Instances instances) {
        // Store actual result
        this.reset(distance, separateDistance, instances);
    }

    ExperimentResult(ArrayList<ExperimentResult> resultList) {
        double actualResult = Double.POSITIVE_INFINITY;
        double[] data = new double[]{};
        Instances instances = null;

        boolean isInf = true;
        for (ExperimentResult result : resultList) {
            isInf = isInf && Double.isInfinite(result.distance);
        }
        if (!isInf){
            // Calculate Mean of actual results and
            // get list of the individual results and their corresponding instances
            data = new double[resultList.get(0).separateDistance.length * resultList.size()];
            instances = new Instances(resultList.get(0).instances, 0);
            actualResult = 0.0f;
            for (int i = 0; i < resultList.size(); i++) {
                ExperimentResult res = resultList.get(i);
                actualResult += res.distance;
                for (int j = 0; j < res.separateDistance.length; j++) {
                    data[i*res.separateDistance.length + j] = res.separateDistance[j];
                    instances.add(res.instances.get(j));
                }
            }
            actualResult /= (double)resultList.size();
        }
        this.reset(actualResult, data, instances);
   }

   public static ExperimentResult mergeExperiments(ArrayList<ExperimentResult> results) {
        if (results.get(0) instanceof StructuredExperimentResult) {
            return new StructuredExperimentResult(results);
        }
        else {
            return new SingleExperimentResult(results);
        }
   }

   private double calcMean() {
        double mean = 0.0f;
        for (double datum : this.separateDistance) {
            mean += datum;
        }
        return mean / this.separateDistance.length;
   }

   private double calcSd() {
        double mean = this.calcMean();
        double sd = 0.0f;
        for (double datum: this.separateDistance) {
            sd = sd + Math.pow(datum - mean, 2);
        }
        sd = sd / (this.separateDistance.length - 1);
        sd = Math.sqrt(sd);
        return sd;
   }

   private int[] calcMinMaxIndex() {
        int[] indexMinMax = new int[]{0,0};
        for (int i = 0; i < this.separateDistance.length; i++) {
            indexMinMax[0] = (this.separateDistance[i] < this.separateDistance[indexMinMax[0]]) ? i : indexMinMax[0];
            indexMinMax[1] = (this.separateDistance[i] > this.separateDistance[indexMinMax[1]]) ? i : indexMinMax[1];
        }
        return indexMinMax;
   }

    private void reset(double actualResult, double[] data, Instances instances) {
        // Store actual result
        this.distance = actualResult;

        if (Double.isInfinite(this.distance)) {
            this.separateDistance = new double[]{};
            this.instances = null;
            this.mean = Double.POSITIVE_INFINITY;
            this.sd = Double.POSITIVE_INFINITY;
            this.maxDist = Double.POSITIVE_INFINITY;
            this.maxInstance = null;
            this.minDist = Double.POSITIVE_INFINITY;
            this.minInstance = null;
        }
        else {
            this.separateDistance = data;
            this.instances = instances;

            this.mean = this.calcMean();
            this.sd = this.calcSd();

            int[] indexMinMax = this.calcMinMaxIndex();
            this.minDist = this.separateDistance[indexMinMax[0]];
            this.minInstance = this.instances.get(indexMinMax[0]);
            this.maxDist = this.separateDistance[indexMinMax[1]];
            this.maxInstance = this.instances.get(indexMinMax[1]);
        }
    }

    public double getDistance() {
        return distance;
    }

    public double getMean() {
        return mean;
    }

    public double getSd() {
        return sd;
    }

    public double getMinDist() {
        return minDist;
    }

    public Instance getMinInstance() {
        return minInstance;
    }

    public double getMaxDist() {
        return maxDist;
    }

    public Instance getMaxInstance() {
        return maxInstance;
    }

    public String[] getSummaryRow() {
        String summaryRow[] = new String[8];

        summaryRow[0] = "";
        for (int i = 0; i < this.instances.get(0).numAttributes(); i++) {
            if (!this.instances.get(0).isMissing(i)) {
                summaryRow[0] += this.instances.attribute(i).name() + "_";
            }
        }
        summaryRow[0] = summaryRow[0].substring(0, summaryRow[0].length() - 1);

        summaryRow[1] = Double.toString(this.distance);
        summaryRow[2] = Double.toString(this.mean);
        summaryRow[3] = Double.toString(this.sd);
        summaryRow[4] = Double.toString(this.maxDist);
        summaryRow[5] = "";
        summaryRow[6] = Double.toString(this.minDist);
        summaryRow[7] = "";

        if (!Double.isInfinite(this.distance)) {
            for (int j = 0; j < this.instances.numAttributes(); j++) {
                if (!this.maxInstance.isMissing(j) && !this.minInstance.isMissing(j)) {
                    String maxVal = this.maxInstance.stringValue(j);
                    summaryRow[5] += this.instances.attribute(j).name() + "=" + maxVal + "_";
                    String minVal = this.minInstance.stringValue(j);
                    summaryRow[7] += this.instances.attribute(j).name() + "=" + minVal + "_";
                }
            }
            // Trim last underscore
            summaryRow[5] = summaryRow[5].substring(0, summaryRow[5].length() - 1);
            summaryRow[7] = summaryRow[7].substring(0, summaryRow[7].length() - 1);
        }
        else {
            summaryRow[5] = "NA";
            summaryRow[7] = "NA";
        }
        return summaryRow;
    }
}
