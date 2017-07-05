package analyse.timeline;

import models.Model;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 20/12/2016.
 */
public class Chunks extends TimelineAnalysis {
    private int chunkAttribute;
    private int chunkSize;

    private List<Integer> storedChunks1;
    private List<Integer> storedChunks2;

    //TODO: add attributes to measure, instead of all attributes, maybe to listener
    public Chunks(int chunkAttribute, int chunkSize, boolean increment, Model referenceModel) {
        this.previousModel = referenceModel.copy();
        this.currentModel = referenceModel.copy();

        this.currentIndex = 0;
        this.chunkAttribute = chunkAttribute;
        this.chunkSize = chunkSize;
        this.increment = increment;

        this.attributeSubsets = this.previousModel.getAllAttributeSubsets();

        this.storedChunks1 = new ArrayList<>();
        this.storedChunks2 = new ArrayList<>();

        this.updateListenerMetadata();
    }

    /**
     * Function to find the indices of the first chunk in the model
     * @param model The model containing all the instances being modeled.
     * @param chunkAttribute The index of the attribute that the chunks are based off.
     * @return The list of increasing indices of the instances in the first chunk
     */
    private static int[] chunkIndices(Model model, int chunkAttribute) {
        ArrayList<Integer> indices = new ArrayList<>();
        int chunkValue = (int)model.getAllInstances().get(0).value(chunkAttribute);
        indices.add(0);
        for (int i = 1; i < model.getAllInstances().size(); i++) {
            if ((int)model.getAllInstances().get(i).value(chunkAttribute) == chunkValue) {
                indices.add(i);
            }
            else {
                break;
            }
        }
        return ArrayUtils.toPrimitive(indices.toArray(new Integer[0]));
    }

     @Override
     public void addInstance(Instance instance) {
        int currentChunk = (int)instance.value(this.chunkAttribute);
        // Check if we are filling the tail/previous model
        if (this.storedChunks1.size() < chunkSize) {
            int tmpSize = this.storedChunks1.size();
            this.previousModel.addInstance(instance);
            if (tmpSize == 0 || this.storedChunks1.get(tmpSize - 1) != currentChunk) {
                this.storedChunks1.add(currentChunk);
            }
            this.currentIndex += 1;
        }
        // Check if we are filling the head/current model
        else if (this.storedChunks2.size() < chunkSize) {
            int tmpSize = this.storedChunks2.size();
            // If current model empty & instance to add belongs to previous chunk
            if (tmpSize == 0 && currentChunk == this.storedChunks1.get(chunkSize - 1)) {
                this.previousModel.addInstance(instance);
                this.currentIndex += 1;
            }
            else {
                this.currentModel.addInstance(instance);
                if (tmpSize == 0 || this.storedChunks2.get(tmpSize - 1) != currentChunk) {
                    this.storedChunks2.add(currentChunk);
                }
            }
        }
        // Both models are filled with the max number of chunks
        else {
            // Current Instance belongs to latest chunk in head model
            if (storedChunks2.get(storedChunks2.size() - 1) == currentChunk) {
                this.currentModel.addInstance(instance);
            }
            // New chunk
            else {
                // First calculate distance and call listeners
                this.addDistance(this.currentIndex);
                // If not analysis is not by increments, the operations are trivial
                if (!this.increment) {
                    this.currentIndex += this.currentModel.size();
                    this.storedChunks1 = this.storedChunks2;
                    this.previousModel = this.currentModel;
                    this.currentModel = this.currentModel.copy();
                    this.storedChunks2 = new ArrayList<>();
                    // Add instance
                    this.currentModel.addInstance(instance);
                    this.storedChunks2.add(currentChunk);
                }
                // Analysis by increments
                else {
                    // Remove oldest chunk in tail model (tail chunk)
                    int[] tailChunkIndices = chunkIndices(this.previousModel, this.chunkAttribute);
                    this.previousModel.removeInstances(tailChunkIndices);
                    this.storedChunks1.remove(0);

                    // Move middle chunk from head to tail model
                    int[] middleChunkIndices = chunkIndices(this.currentModel, this.chunkAttribute);
                    Instances middleInstances = this.currentModel.removeInstances(middleChunkIndices);
                    this.previousModel.addInstances(middleInstances);
                    this.storedChunks1.add(this.storedChunks2.get(0));
                    this.storedChunks2.remove(0);

                    // Add new chunk to head model
                    this.currentModel.addInstance(instance);
                    this.storedChunks2.add((int)instance.value(this.chunkAttribute));

                    this.currentIndex += middleInstances.size();
                }
            }
        }
     }

}
