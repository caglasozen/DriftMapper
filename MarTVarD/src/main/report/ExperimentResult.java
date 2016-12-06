package main.report;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
**/

public class ExperimentResult {
    private double mean;
    private double sd;
    private double maxDist;
    private Instance maxInstance;
    private double minDist;
    private Instance minInstance;
    private double distance;

    private double[] separateDistance;
    private Instances instances;

    public ExperimentResult(double distance, double[] separateDistance, Instances instances) {
        // Store actual result
        this.reset(distance, separateDistance, instances);
    }

    public ExperimentResult(ArrayList<ExperimentResult> resultList) {
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

   public ExperimentResult(double weightAverageDistance, double[] separateDistance, Instances instances,
                           ArrayList<ExperimentResult> separateExperiments, ArrayList<Double> experimentsProbability) {
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
        summaryRow[0] = Double.toString(this.distance);
        summaryRow[1] = Double.toString(this.mean);
        summaryRow[2] = Double.toString(this.sd);
        summaryRow[3] = Double.toString(this.maxDist);
        summaryRow[4] = "";
        summaryRow[5] = Double.toString(this.minDist);
        summaryRow[6] = "";
        summaryRow[7] = "";
        for (int i = 0; i < this.instances.get(0).numAttributes(); i++) {
            if (!this.instances.get(0).isMissing(i)) {
                summaryRow[7] += this.instances.attribute(i).name() + "_";
            }
        }
        summaryRow[7] = summaryRow[7].substring(0, summaryRow[7].length() - 1);
        if (!Double.isInfinite(this.distance)) {
            for (int j = 0; j < this.instances.numAttributes(); j++) {
                if (!this.maxInstance.isMissing(j) && !this.minInstance.isMissing(j)) {
                    String maxVal = this.maxInstance.stringValue(j);
                    summaryRow[4] += this.instances.attribute(j).name() + "=" + maxVal + "_";
                    String minVal = this.minInstance.stringValue(j);
                    summaryRow[6] += this.instances.attribute(j).name() + "=" + minVal + "_";
                }
            }
            // Trim last underscore
            summaryRow[4] = summaryRow[4].substring(0, summaryRow[4].length() - 1);
            summaryRow[6] = summaryRow[6].substring(0, summaryRow[6].length() - 1);
        }
        else {
            summaryRow[4] = "NA";
            summaryRow[6] = "NA";
        }
        return summaryRow;
    }
}
