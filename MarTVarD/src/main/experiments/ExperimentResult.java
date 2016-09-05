package main.experiments;

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

    public ExperimentResult(double actualResult, double[] data, double[][] instanceValues) {
        // Store actual result
        this.actualResult = actualResult;
        if (Double.isInfinite(this.actualResult)) {
            this.mean = Double.POSITIVE_INFINITY;
            this.sd = Double.POSITIVE_INFINITY;
            this.maxDist = Double.POSITIVE_INFINITY;
            this.maxValues = new double[]{};
            this.minDist = Double.POSITIVE_INFINITY;
            this.minValues = new double[]{};
        }
        else {
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
