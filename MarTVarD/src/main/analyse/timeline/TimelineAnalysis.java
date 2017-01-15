package main.analyse.timeline;

import com.opencsv.CSVWriter;
import main.models.Model;
import weka.core.Instance;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Map;

import main.DriftMeasurement;
import weka.core.Instances;

/**
 * Created by loongkuan on 2/12/2016.
 */
public abstract class TimelineAnalysis {
    protected Model previousAllModel;
    protected Model currentAllModel;

    protected Model previousModel;
    protected Model currentModel;

    protected DriftMeasurement[] driftMeasurementTypes;
    protected int currentIndex;

    protected Map<DriftMeasurement, ArrayList<Integer>> driftPoints;
    protected Map<DriftMeasurement, ArrayList<double[]>> driftValues;
    protected ArrayList<int[]> attributeSubsets;

    public abstract void addInstance(Instance instance);

    protected void addDistance(int driftPoint) {
        for (DriftMeasurement type : this.driftMeasurementTypes) {
            double[] allDist = new double[this.attributeSubsets.size() + 1];
            allDist[0] = this.getFullDistance(type);
            for (int i = 0; i < this.attributeSubsets.size(); i++) {
                double dist = this.getDistance(this.attributeSubsets.get(i), type);
                allDist[1 + i] = dist;
            }
            ArrayList<Integer> currentDriftPoints = this.driftPoints.get(type);
            currentDriftPoints.add(driftPoint);
            ArrayList<double[]> currentDriftValues = this.driftValues.get(type);
            currentDriftValues.add(allDist);

            this.driftPoints.put(type, currentDriftPoints);
            this.driftValues.put(type, currentDriftValues);
        }
    }

    protected double getDistance(int[] attributeSubset, DriftMeasurement driftMeasurementType) {
        switch (driftMeasurementType) {
            case COVARIATE:
                return this.previousModel.peakCovariateDistance(this.currentModel, attributeSubset, 1.0);
            case JOINT:
                return this.previousModel.peakJointDistance(this.currentModel, attributeSubset, 1.0);
            case LIKELIHOOD:
                return this.previousModel.peakLikelihoodDistance(this.currentModel, attributeSubset, 1.0);
            case POSTERIOR:
                return this.previousModel.peakPosteriorDistance(this.currentModel, attributeSubset, 1.0);
        }
        return Double.NaN;
    }

   protected double getFullDistance(DriftMeasurement driftMeasurementType) {
        switch (driftMeasurementType) {
            case COVARIATE:
                return this.previousAllModel.peakCovariateDistance(this.currentAllModel,
                        this.previousAllModel.getAttributesAvailable(), 1.0);
            case JOINT:
                return this.previousAllModel.peakJointDistance(this.currentAllModel,
                        this.previousAllModel.getAttributesAvailable(), 1.0);
            case LIKELIHOOD:
                return this.previousAllModel.peakLikelihoodDistance(this.currentAllModel,
                        this.previousAllModel.getAttributesAvailable(), 1.0);
            case POSTERIOR:
                return this.previousAllModel.peakPosteriorDistance(this.currentAllModel,
                        this.previousAllModel.getAttributesAvailable(), 1.0);
        }
        return Double.NaN;
    }

    public void writeResultsToFile(String filename, DriftMeasurement driftMeasurement) {
        ArrayList<Integer> currentDriftPoints = this.driftPoints.get(driftMeasurement);
        ArrayList<double[]> currentDriftValues = this.driftValues.get(driftMeasurement);

        ArrayList<String> headerList = new ArrayList<>();
        headerList.add("points");
        headerList.add("all_attributes");
        for (int i = 0; i < this.attributeSubsets.size(); i++) {
            String tmp = "";
            for (int attributeIndex : this.attributeSubsets.get(i)) {
                tmp += this.previousModel.getAllInstances().attribute(attributeIndex).name() + "_";
            }
            // Trim last underscore
            tmp = tmp.substring(0, tmp.length() - 1);
            headerList.add(tmp);
        }
        String[] header = headerList.toArray(new String[headerList.size()]);

        try {
            CSVWriter writer = new CSVWriter(new FileWriter(filename), ',');
            // feed in your array (or convert your data to an array)
            writer.writeNext(header);
            for (int i = 0; i < currentDriftPoints.size(); i++) {
                String[] row = new String[currentDriftValues.get(i).length + 1];
                row[0] = Integer.toString(currentDriftPoints.get(i));
                for (int j = 0; j < currentDriftValues.get(i).length; j++) {
                    row[1 + j] = Double.toString(currentDriftValues.get(i)[j]);
                }
                writer.writeNext(row);
            }
            writer.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }

    }
}
