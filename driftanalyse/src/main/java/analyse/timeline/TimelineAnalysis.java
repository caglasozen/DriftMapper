package analyse.timeline;

import com.opencsv.CSVWriter;
import models.Model;
import result.timeline.TimelineListener;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import global.DriftMeasurement;

/**
 * Created by loongkuan on 2/12/2016.
 */
public abstract class TimelineAnalysis {
    private List<TimelineListener> listeners = new ArrayList<>();

    protected Model previousModel;
    protected Model currentModel;

    protected DriftMeasurement[] driftMeasurementTypes;

    protected int currentIndex;

    protected Map<DriftMeasurement, ArrayList<Integer>> driftPoints;
    protected Map<DriftMeasurement, ArrayList<double[]>> driftValues;
    protected ArrayList<int[]> attributeSubsets;

    public abstract void addInstance(Instance instance);

    public int getCurrentIndex() {
        return currentIndex;
    }

    public void addListener(TimelineListener toAdd) {
        this.listeners.add(toAdd);
    }

    public void updateListenerMetadata() {
        String[] header = this.getAttributeSubsetHeader();
        for (TimelineListener listener : this.listeners) {
            listener.updateMetaData(header);
        }
    }

    public void addDistance(int driftPoint) {
        for (TimelineListener listener : this.listeners) {
            double[] allDist = new double[this.attributeSubsets.size()];
            for (int i = 0; i < this.attributeSubsets.size(); i++) {
                double dist = this.getDistance(this.attributeSubsets.get(i), listener.getMeasurementType());
                allDist[i] = dist;
            }
            listener.returnDriftPointMagnitude(driftPoint, allDist);
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
            case CLASS:
                return this.previousModel.peakClassDistance(this.currentModel, 1.0);
        }
        return Double.NaN;
    }

    public String[] getAttributeSubsetHeader() {
        ArrayList<String> headerList = new ArrayList<>();
        for (int i = 0; i < this.attributeSubsets.size(); i++) {
            String tmp = "";
            for (int attributeIndex : this.attributeSubsets.get(i)) {
                tmp = tmp.concat(this.previousModel.getAllInstances().attribute(attributeIndex).name() + "_");
            }
            // Trim last underscore
            tmp = tmp.substring(0, tmp.length() - 1);
            headerList.add(tmp);
        }
        return headerList.toArray(new String[headerList.size()]);
    }

    public void writeResultsToFile(String filename, DriftMeasurement driftMeasurement) {
        ArrayList<Integer> currentDriftPoints = this.driftPoints.get(driftMeasurement);
        ArrayList<double[]> currentDriftValues = this.driftValues.get(driftMeasurement);

        String[] header = ArrayUtils.addAll(new String[]{"points"}, this.getAttributeSubsetHeader());
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
