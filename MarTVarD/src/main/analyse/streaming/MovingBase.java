package main.analyse.streaming;

import main.models.Model;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by loongkuan on 29/11/2016.
 */
public class MovingBase extends StreamAnalysis {
    private int maxWindowSize;
    private double thresholdDrift;
    private Model referenceModel;
    private double sampleScale = 1.0;

    private Instances currentWindow;
    private boolean baseStable;
    private Model headModel;
    private Model tailModel;
    private int totalInstancesPast;
    HoeffdingAdaptiveTree tree = new HoeffdingAdaptiveTree();

    private Map<Integer, Double> driftPoints;


    public MovingBase(int maxWindowSize, double thresholdDrift, Model referenceModel) {
        this.maxWindowSize = maxWindowSize;
        this.thresholdDrift = thresholdDrift;
        this.referenceModel = referenceModel;
        this.currentWindow = referenceModel.getAllInstances();

        this.reset();
    }

    public void reset() {
        this.baseStable = false;
        this.currentWindow = new Instances(this.currentWindow, this.currentWindow.size());
        this.tailModel = this.referenceModel.copy();
        this.headModel = this.referenceModel.copy();
        this.driftPoints = new HashMap<>();
        this.totalInstancesPast = 0;
    }

    public void addInstance(Instance instance) {
        // Add instance
        this.headModel.addInstance(instance);
        if (this.currentWindow.size() % 2 == 1 || this.currentWindow.size() > this.maxWindowSize) {
            this.headModel.removeInstance(0);
            this.tailModel.addInstance(this.currentWindow.get(this.tailModel.size()));
        }
        if (this.currentWindow.size() > this.maxWindowSize) {
            this.tailModel.removeInstance(0);
            this.currentWindow.remove(0);
        }
        this.currentWindow.add(instance);
        this.updateDetector();
    }

    private void updateDetector() {
        if (baseStable) {
            baseStable = false;
            //checkDrift();
        }
        else {
            // TODO: Change to a better representation of distance
            //ExperimentResult result = this.tailModel.findJointDistance(this.headModel,
            //this.tailModel.getAttributesAvailable(), 1);
            double dist = this.tailModel.peakJointDistance(this.headModel, this.tailModel.getAttributesAvailable(), 1);
            if (dist < this.thresholdDrift/2 && this.currentWindow.size() > this.maxWindowSize/2) {
                this.baseStable = true;
            }
        }
    }

    private void checkDrift() {
        if (this.tailModel.findJointDistance(this.headModel,
                this.tailModel.getAttributesAvailable(), this.sampleScale).getDistance() > this.thresholdDrift) {
            this.reduceDrift();
        }
    }

    private void reduceDrift() {
        double drift = this.tailModel.findJointDistance(this.headModel,
                this.tailModel.getAttributesAvailable(), this.sampleScale).getDistance();
        double originalDrift = drift;
        System.out.println("\rDrift of " + originalDrift + " at " + this.totalInstancesPast);
        while (drift > 0.25) {
            if (this.currentWindow.size() % 2 == 1) {
                this.headModel.removeInstance(this.tailModel.size());
                this.tailModel.addInstance(this.currentWindow.get(this.tailModel.size()));
            }
            this.tailModel.removeInstance(0);
            this.currentWindow.remove(0);
            this.totalInstancesPast += 1;
            drift = this.tailModel.findJointDistance(this.headModel,
                    this.tailModel.getAttributesAvailable(), this.sampleScale).getDistance();
        }
        this.driftPoints.put(this.totalInstancesPast, originalDrift);
        this.baseStable = false;
    }

    public void printDriftPoints() {
        for (int point : this.driftPoints.keySet()) {
            System.out.println("Drift of " + this.driftPoints.get(point) + " at " + point);
        }
    }
}
