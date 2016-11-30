package main.analyse;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
**/

public class ExperimentResult {
    public final double mean;
    public final double sd;
    public final double maxDist;
    public final double[] maxValues;
    public final double minDist;
    public final double[] minValues;
    public final double actualResult;

    private final double[] data;
    private final double[][] instanceValues;

    // TODO: Refactor
    public ExperimentResult(ArrayList<ExperimentResult> resultList) {
        boolean isInf = true;
        for (int i = 0; i < resultList.size(); i++) {
            isInf = isInf && Double.isInfinite(resultList.get(i).actualResult);
        }

        if (isInf) {
            this.actualResult = Double.POSITIVE_INFINITY;
            this.data = new double[]{};
            this.instanceValues = new double[][]{};
            this.mean = Double.POSITIVE_INFINITY;
            this.sd = Double.POSITIVE_INFINITY;
            this.maxDist = Double.POSITIVE_INFINITY;
            this.maxValues = new double[]{};
            this.minDist = Double.POSITIVE_INFINITY;
            this.minValues = new double[]{};
        }
        else {
            // Calculate Mean of actual results
            double tmpRes = 0.0f;
            for (ExperimentResult result : resultList) {
                tmpRes += result.actualResult;
            }
            this.actualResult = tmpRes / (double) resultList.size();

            ArrayList<Double> tmpData = new ArrayList<>();
            ArrayList<double[]> tmpInst = new ArrayList<>();
            for (int i = 0; i < resultList.size(); i++) {
                ExperimentResult res = resultList.get(i);
                for (int j = 0; j < res.data.length; j++) {
                    tmpData.add(res.data[j]);
                    tmpInst.add(res.instanceValues[j]);
                }
            }

            this.data = new double[tmpData.size()];
            this.instanceValues = new double[tmpData.size()][tmpInst.get(0).length];
            for (int i = 0; i < tmpData.size(); i++) {
                this.data[i] = tmpData.get(i);
                this.instanceValues[i] = tmpInst.get(i);
            }

            // Calculate Mean
            double sum = 0.0f;
            for (double datum : data) {
                sum += datum;
            }
            this.mean = sum / data.length;

            // Calculate tmpSd
            double tmpSd = 0.0f;
            for (ExperimentResult res : resultList) {
                tmpSd = tmpSd + Math.pow(res.actualResult - this.actualResult, 2);
            }
            tmpSd = tmpSd / (resultList.size() - 1);
            tmpSd = Math.sqrt(tmpSd);
            this.sd = tmpSd;

            int minIndex = 0;
            int maxIndex = 0;
            for (int i = 0; i < data.length; i++) {
                minIndex = (data[i] < data[minIndex]) ? i : minIndex;
                maxIndex = (data[i] > data[maxIndex]) ? i : maxIndex;
            }
            // Get minimum
            this.minValues = instanceValues[minIndex];
            this.minDist = data[minIndex];

            // Get maximum
            this.maxValues = instanceValues[maxIndex];
            this.maxDist = data[maxIndex];
        }
   }

    public ExperimentResult(double actualResult, double[] data, double[][] instanceValues) {
        // Store actual result
        this.actualResult = actualResult;

        if (Double.isInfinite(this.actualResult)) {
            this.data = new double[]{};
            this.instanceValues = new double[][]{};
            this.mean = Double.POSITIVE_INFINITY;
            this.sd = Double.POSITIVE_INFINITY;
            this.maxDist = Double.POSITIVE_INFINITY;
            this.maxValues = new double[]{};
            this.minDist = Double.POSITIVE_INFINITY;
            this.minValues = new double[]{};
        }
        else {
            this.data = data;
            this.instanceValues = instanceValues;

            // Calculate Mean
            double sum = 0.0f;
            for (double datum : data) {
                sum += datum;
            }
            this.mean = sum / data.length;

            // Calculate tmpSd
            double tmpSd = 0.0f;
            for (double datum: data) {
                tmpSd = tmpSd + Math.pow(datum - this.mean, 2);
            }
            tmpSd = tmpSd / (data.length - 1);
            tmpSd = Math.sqrt(tmpSd);
            this.sd = tmpSd;

            int minIndex = 0;
            int maxIndex = 0;
            for (int i = 0; i < data.length; i++) {
                minIndex = (data[i] < data[minIndex]) ? i : minIndex;
                maxIndex = (data[i] > data[maxIndex]) ? i : maxIndex;
            }
            // Get minimum
            this.minValues = instanceValues[minIndex];
            this.minDist = data[minIndex];

            // Get maximum
            this.maxValues = instanceValues[maxIndex];
            this.maxDist = data[maxIndex];
        }
    }
}
