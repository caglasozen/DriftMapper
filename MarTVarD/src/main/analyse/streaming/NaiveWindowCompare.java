package main.analyse.streaming;

import com.opencsv.CSVWriter;
import main.DriftMeasurement;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import weka.core.Instance;

import java.io.FileWriter;
import java.util.ArrayList;

/**
 * Created by loongkuan on 14/12/2016.
 */
public class NaiveWindowCompare extends StreamAnalysis {
    //TODO: Be able to measure different types of drift and name accordingly
    private int windowSize;
    private DriftMeasurement driftMeasurementType;
    private ArrayList<int[]> attributeSubsets;
    private Model previousModel;
    private Model currentModel;
    private Model previousAllModel;
    private Model currentAllModel;

    private int currentIndex;
    private ArrayList<Integer> driftPoints;
    private ArrayList<double[]> driftValues;

    public NaiveWindowCompare(int windowSize, Model referenceModel,
                              ArrayList<int[]> attributeSubsets, DriftMeasurement driftMeasurementType) {
        this.windowSize = windowSize;
        this.driftMeasurementType = driftMeasurementType;
        this.currentModel = referenceModel.copy();
        this.currentAllModel = referenceModel.copy();
        ((FrequencyMaps)currentAllModel).changeAttributeSubsetLength(this.currentModel.getAllInstances().numAttributes() - 1);
        this.attributeSubsets = attributeSubsets;

        this.currentIndex = -1;
        this.driftPoints = new ArrayList<>();
        this.driftValues = new ArrayList<>();
    }

    @Override
    public void reset() {

    }

    @Override
    public void addInstance(Instance instance) {
        this.currentIndex += 1;
        if (currentModel.size() < this.windowSize) {
            this.currentModel.addInstance(instance);
            this.currentAllModel.addInstance(instance);
        }
        else {
            if (this.previousModel != null) {
                double[] allDist = new double[this.attributeSubsets.size() + 1];
                allDist[0] = this.getFullDistance();
                for (int i = 0; i < this.attributeSubsets.size(); i++) {
                    double dist = this.getDistance(this.attributeSubsets.get(i));
                    allDist[1 + i] = dist;
                }
                this.driftPoints.add(this.currentIndex - this.windowSize);
                this.driftValues.add(allDist);
            }
            this.previousModel = this.currentModel;
            this.currentModel = this.currentModel.copy();
            this.currentModel.addInstance(instance);

            this.previousAllModel = this.currentAllModel;
            this.currentAllModel= this.currentAllModel.copy();
            this.currentAllModel.addInstance(instance);
        }
    }

    public void writeResultsToFile(String filename) {
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
            for (int i = 0; i < this.driftPoints.size(); i++) {
                String[] row = new String[this.driftValues.get(i).length + 1];
                row[0] = Integer.toString(this.driftPoints.get(i));
                for (int j = 0; j < this.driftValues.get(i).length; j++) {
                    row[1 + j] = Double.toString(this.driftValues.get(i)[j]);
                }
                writer.writeNext(row);
            }
            writer.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private double getDistance(int[] attributeSubset) {
        switch (this.driftMeasurementType) {
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

   private double getFullDistance() {
        switch (this.driftMeasurementType) {
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
}
