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

    private Model modelCurrent;
    private double maxDriftToIgnore;
    private boolean isDrifting;

    private Model[] currentDetectorModels;
    private int[] currentInterval;
    private int currentIndex;
    private double currentDrift;

    private ArrayList<int[]> intervals;
    private ArrayList<Double> drifts;



    public MovingBase(Model referenceModel) {
        this.referenceModel = referenceModel;
        /*
        this.maxWindowSize = maxWindowSize;
        this.thresholdDrift = thresholdDrift;
        this.currentWindow = referenceModel.getAllInstances();
        */

        this.reset();
    }

    public void reset() {
        /*
        this.baseStable = false;
        this.currentWindow = new Instances(this.currentWindow, this.currentWindow.size());
        this.tailModel = this.referenceModel.copy();
        this.headModel = this.referenceModel.copy();
        this.driftPoints = new HashMap<>();
        this.totalInstancesPast = 0;
        */

        // TODO: Make params
        this.resolution = 10000;
        this.modelCurrent = this.referenceModel.copy();
        this.maxDriftToIgnore = 0.1;
        this.intervals = new ArrayList<>();
        this.drifts = new ArrayList<>();
        this.isDrifting = false;
        this.currentIndex = 0;
        this.currentDrift = 0.0;
    }

    public void addInstance(Instance instance) {
        this.modelCurrent.addInstance(instance);
        this.currentIndex += 1;
        if (this.modelCurrent.size() % this.resolution == 0) {
            if (this.currentDetectorModels == null) {
                this.currentDetectorModels = new Model[2];
                this.currentDetectorModels[0] = this.modelCurrent;
                this.modelCurrent = this.referenceModel.copy();

                this.currentInterval = new int[]{this.currentIndex - this.resolution, this.currentIndex};
            }
            else {
                this.updateDetector();
            }
        }
    }

    private void updateDetector() {
        this.currentInterval[1] = this.currentIndex;
        double dist = this.currentDetectorModels[0].peakPosteriorDistance(this.modelCurrent,
                this.modelCurrent.getAttributesAvailable(), this.sampleScale);
        //System.out.println("\r" + dist + " at " + this.currentIndex);
        if (!this.isDrifting) {
            if (dist > this.maxDriftToIgnore) {
                this.isDrifting = true;
                this.currentDrift = dist;
                this.currentDetectorModels[1] = this.modelCurrent;
            }
            else {
                this.currentInterval[0] = currentIndex;
                this.currentDetectorModels[0].addInstances(this.modelCurrent.getAllInstances());
            }
        }
        else {
            // Larger drift detected, continue moving window
            if (this.currentDrift < dist + 0.001) {
                // Add all instances in current model to old one to try and maximise drift
                this.currentDetectorModels[1].addInstances(this.modelCurrent.getAllInstances());
                double dist2 = this.currentDetectorModels[0].peakPosteriorDistance(this.currentDetectorModels[1],
                        this.currentDetectorModels[1].getAttributesAvailable(), this.sampleScale);
                if (dist2 < dist) {
                    this.currentDrift = dist;
                    this.currentDetectorModels[1] = this.modelCurrent;
                }
                else {
                    this.currentDrift = dist2;
                }
            }
            else {
                this.intervals.add(this.currentInterval);
                this.drifts.add(this.currentDrift);
                this.isDrifting = false;
                this.currentDrift = 0;
                this.currentDetectorModels[0] = this.modelCurrent;
                this.currentDetectorModels[1] = null;
                this.currentInterval = new int[]{this.currentIndex, this.currentIndex};
            }
        }
        this.modelCurrent = this.referenceModel.copy();
    }

    private void checkDrift() {
        if (this.tailModel.findCovariateDistance(this.headModel,
                this.tailModel.getAttributesAvailable(), this.sampleScale).getDistance() > this.thresholdDrift) {
            this.reduceDrift();
        }
    }

    private void reduceDrift() {
        double drift = this.tailModel.findCovariateDistance(this.headModel,
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
            drift = this.tailModel.findPosteriorDistance(this.headModel,
                    this.tailModel.getAttributesAvailable(), this.sampleScale).getDistance();
        }
        this.driftPoints.put(this.totalInstancesPast, originalDrift);
        this.baseStable = false;
    }

    public void printDriftPoints() {
        for (int i = 0; i < this.intervals.size(); i++) {
            System.out.println("Drift of " + this.drifts.get(i) +
                    " from " + this.intervals.get(i)[0] + " to " + this.intervals.get(i)[1]);
        }
    }
}
