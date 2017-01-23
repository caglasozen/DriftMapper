package main.analyse.timeline;

import main.DriftMeasurement;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Created by loongkuan on 14/12/2016.
 */
public class NaiveMovingWindow extends TimelineAnalysis{
    //TODO: Be able to measure different types of drift and name accordingly
    private int windowSize;

    public NaiveMovingWindow(int windowSize, Model referenceModel, DriftMeasurement[] driftMeasurementType) {
        this.windowSize = windowSize;
        this.driftMeasurementTypes = driftMeasurementType;
        this.previousModel= referenceModel.copy();
        this.previousAllModel= referenceModel.copy();
        this.currentModel = referenceModel.copy();
        this.currentAllModel = referenceModel.copy();
        currentAllModel.changeAttributeSubsetLength(referenceModel.getAttributesAvailable().length);
        previousAllModel.changeAttributeSubsetLength(referenceModel.getAttributesAvailable().length);
        this.attributeSubsets = referenceModel.getAllAttributeSubsets();

        this.currentIndex = -1;

        this.driftPoints = new HashMap<>();
        this.driftValues = new HashMap<>();
        for (DriftMeasurement type : this.driftMeasurementTypes) {
            this.driftPoints.put(type, new ArrayList<>());
            this.driftValues.put(type, new ArrayList<>());
        }
    }

    public void addInstance(Instance instance) {
        this.currentIndex += 1;
        if (previousModel.size() < this.windowSize) {
            this.previousModel.addInstance(instance);
            this.previousAllModel.addInstance(instance);
        }
        else if (currentModel.size() < this.windowSize) {
            this.currentModel.addInstance(instance);
            this.currentAllModel.addInstance(instance);
        }
        else {
            this.previousModel.removeInstance(0);
            this.previousModel.addInstance(this.currentModel.getAllInstances().get(0));
            this.currentModel.removeInstance(0);
            this.currentModel.addInstance(instance);

            this.previousAllModel.removeInstance(0);
            this.previousAllModel.addInstance(this.currentModel.getAllInstances().get(0));
            this.currentAllModel.removeInstance(0);
            this.currentAllModel.addInstance(instance);
        }
        if (previousModel.size() >= windowSize && currentModel.size() >= windowSize) {
            this.addDistance(this.currentIndex - this.windowSize);
        }
    }
}
