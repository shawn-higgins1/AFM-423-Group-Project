�	�hq�0�6@�hq�0�6@!�hq�0�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�hq�0�6@�Վ�u4@1J�y��?AA��ǘ��?I s-Z�� @*	433333G@2U
Iterator::Model::ParallelMapV2�~j�t��?!	�=���9@)�~j�t��?1	�=���9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!���˽=@)g��j+��?1�rO#,79@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_�L�?!�FX�i6@)9��v��z?1�FX�i,@:Preprocessing2F
Iterator::Model��y�):�?!��FX.C@)�����w?1      )@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!,�4�r� @)�q����o?1,�4�r� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!|a���N@)�~j�t�h?1	�=���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!���{@)�J�4a?1���{@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!��FXn9@)Ǻ���V?1�4�rO#@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL?�j�X@Q�,�H%|�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Վ�u4@�Վ�u4@!�Վ�u4@      ��!       "	J�y��?J�y��?!J�y��?*      ��!       2	A��ǘ��?A��ǘ��?!A��ǘ��?:	 s-Z�� @ s-Z�� @! s-Z�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qL?�j�X@y�,�H%|�?�"5
sequential/dense/MatMulMatMul�
���?!�
���?0"C
%gradient_tape/sequential/dense/MatMulMatMul=Bn�͠?!a&���ڱ?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul=Bn�͠?!�G솈A�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul=Bn�͠?!O��D6T�?"7
sequential/dense_1/MatMulMatMul=Bn�͠?!�D�E���?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�
�F�?!��!�'�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�҉:�?!nb��?"!
Adam/PowPow=Bn�͐?!��-c���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad=Bn�͐?!�R�c���?"7
sequential/dense_2/MatMulMatMul=Bn�͐?!��$����?0Q      Y@Y>����/@aX�i��U@qq�L~�X@yG솈A�?"�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 