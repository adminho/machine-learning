from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim
from nets import utils
from nets import resnet_v1
import numpy as np
import sys


#*************************************************************
#
#	build the whole Supervised Pyramid Context Network(SPCNET)
#		
#*************************************************************
def build_SPC(inputs, config, is_training, backbone='resnet50'):
	# Parse the inputs
	input_image = inputs['input_image']
	input_image = input_image - config.MEAN_PIXEL
	image_shape = config.IMAGE_SHAPE
	if is_training:
		# RPN GT
		input_rpn_match = inputs['input_rpn_match']
		input_rpn_bbox = inputs['input_rpn_bbox']
		# Detection GT (class IDs, bounding boxes, and masks)
		# 1. GT Class IDs (zero padded)
		input_gt_class_ids = inputs['input_gt_class_ids']
		# 2. GT Boxes in pixels (zero padded)
		# [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in Normalize coordinates
		input_gt_boxes = inputs["input_gt_boxes"]
		input_gt_boxes = utils.norm_boxes_graph(input_gt_boxes, tf.shape(input_image)[1:3])
		# 3. GT Masks (zero padded)
		# [batch, MAX_GT_INSTANCES, height, width]
		input_gt_masks = inputs['input_gt_masks']
		# SPCNET gloabel text segmentation
		input_gt_global_masks = inputs['input_gt_global_masks']
	
   	# pyramid_feature Dict{P2, P3, P4, P5} of feature maps from different level of the
	# feature pyramid. Each is [batch, height, width, channels]
	pyramid_feature = build_FPN(input_image, config, is_training, backbone)
	# get the pyramid feature maps shape
	fpn_shapes = []
	for i in range(2, 6, 1):
		p = 'P%d' % i
		shape = pyramid_feature[p].shape
		fpn_shapes.append([shape[1], shape[2]])
	fpn_shapes = np.array(fpn_shapes)
	# get global text segmentation map and saliency map from per pyramid_feature
	gts, tcm_outputs = build_TCM(pyramid_feature, image_shape, config)
	# get all anchors
	anchors = generate_all_anchors(fpn_shapes, image_shape,config)
	# number of anchors per pixel in the feature map
	anchors_num = len(config.RPN_ANCHOR_RATIOS)
	# build rpn model and get outputs
	rpn_class_logits, rpn_prob, rpn_bbox = build_RPN(tcm_outputs, image_shape, anchors_num, is_training, config)
	# Generate proposals
	# Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
   	# and zero padded.
	proposal_count = config.POST_NMS_ROIS_TRAINING if is_training\
					 else config.POST_NMS_ROIS_INFERENCE
	rpn_rois = generate_proposal(rpn_prob, rpn_bbox, anchors, proposal_count, config)
	assert config.USE_RPN_ROIS == True, "Don't use rpn rois not implement"
	
	if is_training:
		# Generate detection targets
		# Subsamples proposals and generates target outputs for training
		# Note that proposal class IDs, gt_boxes, and gt_masks are zero
		# padded. Equally, returned rois and targets are zero padded.
		rois, target_class_ids, target_bbox, target_mask = generate_detect_target(rpn_rois,\
								input_gt_class_ids, input_gt_boxes, input_gt_masks, config)
		# Network Heads
		# TODO: verify that this handles zero padded ROIs
		mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = build_mrcnn_head(rois, tcm_outputs,\
								image_shape, is_training, config)
		mrcnn_mask_logits, mrcnn_mask = build_mrcnn_mask(rois, tcm_outputs,\
								image_shape, is_training, config)
		# loss
		rpn_class_loss = build_rpn_class_loss(input_rpn_match, rpn_class_logits, config)
		rpn_bbox_loss = build_rpn_bbox_loss(input_rpn_bbox, input_rpn_match, rpn_bbox, config)
		mrcnn_class_loss = build_mrcnn_class_loss(target_class_ids, mrcnn_class_logits, config)
		mrcnn_bbox_loss = build_mrcnn_bbox_loss(target_bbox, target_class_ids, mrcnn_bbox, config)
		mrcnn_mask_loss = build_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask_logits, config)
		global_mask_loss = build_global_mask_loss(input_gt_global_masks, gts, config)

		losses = {}
		losses['rpn_class_loss'] = rpn_class_loss * config.LOSS_WEIGHTS['rpn_class_loss']
		losses['rpn_bbox_loss'] = rpn_bbox_loss * config.LOSS_WEIGHTS['rpn_bbox_loss']
		losses['mrcnn_class_loss'] = mrcnn_class_loss * config.LOSS_WEIGHTS['mrcnn_class_loss']
		losses['mrcnn_bbox_loss'] = mrcnn_bbox_loss * config.LOSS_WEIGHTS['mrcnn_bbox_loss']
		losses['mrcnn_mask_loss'] = mrcnn_mask_loss * config.LOSS_WEIGHTS['mrcnn_mask_loss']
		losses['global_mask_loss'] = global_mask_loss * config.LOSS_WEIGHTS['global_mask_loss']
		losses['total_loss'] = tf.add_n([losses[k] for k in losses.keys()] +
								tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return losses
	else:
		# Network Heads
		# Proposal classifier and BBox regressor heads
		mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = build_mrcnn_head(rpn_rois, tcm_outputs,\
								image_shape, is_training, config)
		# Create masks for detections
		mrcnn_mask_logits, mrcnn_mask = build_mrcnn_mask(rpn_rois, tcm_outputs,\
								image_shape, is_training, config)
		# Reshape global text segmentation map
		# Original: Dict{P2, P3, P4, P5} each is [batch, H, W, NUM_CLASSES]
		# Reshape:  [Batch, 4, H, W, NUM_CLASSES]
		stack_gts = []
		for i in range(config.BATCH_SIZE):
			stack_gts.append(tf.stack([gts[P][i,...] for P in ['P2', 'P3', 'P4', 'P5']],\
							axis=0))
		gts = tf.stack(stack_gts, axis=0)
		# Detections
		# output is 
		# detection :[batch, num_detections, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
		# mask: [batch, num_detections, MASK_H, MASK_W, class_num]
		detections, masks = get_detect_results(rpn_rois, mrcnn_prob, 
								mrcnn_bbox, mrcnn_mask, gts, config)

		return detections, masks


def build_input_graph(is_training, config):
	"""
	build input tensors
	"""
	inputs = {}
	# Image size must be dividable by 2 multiple times
	h, w = config.IMAGE_SHAPE[:2]
	if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
		raise Exception("Image size must be dividable by 2 at least 6 times "\
						"to avoid fractions when downscaling and upscaling."\
						"For example, use 256, 320, 384, 448, 512, ... etc. ")

	inputs['input_image'] = tf.placeholder(tf.float32,\
							shape=[None, config.IMAGE_SHAPE[0],\
							config.IMAGE_SHAPE[1], 3],\
							name='input_image')
	if is_training:
		#RPN GT
		inputs['input_rpn_match'] = tf.placeholder(tf.int32,\
							shape=[None, None, 1], name='input_rpn_match')
		inputs['input_rpn_bbox'] = tf.placeholder(tf.float32,\
							shape=[None, None, 4], name='input_rpn_bbox')
		# Detection GT (class IDs, bounding boxes, and masks)
		# 1. GT Class IDs (zero padded)
		inputs['input_gt_class_ids'] = tf.placeholder(tf.int32,\
							shape=[None, None], name="input_gt_class_ids")
		# 2. GT Boxes in pixels (zero padded)
		# [batch, MAX_GT_INSTANCES * len(gpu_list), (y1, x1, y2, x2)] 
		# in normalized coordinates
		inputs['input_gt_boxes'] = tf.placeholder(tf.float32,\
							shape=[None, None, 4], name="input_gt_boxes")
		# 3. GT Masks (zero padded)
		# [batch, height, width, MAX_GT_INSTANCES * len(gpu_list)]
		if config.USE_MINI_MASK:
			inputs['input_gt_masks'] = tf.placeholder(tf.bool,\
							shape=[None, config.MINI_MASK_SHAPE[0],\
							config.MINI_MASK_SHAPE[1], None], name='input_gt_masks')
		else:
			inputs['input_gt_masks'] = tf.placeholder(tf.bool,\
							shape=[None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],\
							name='input_gt_masks')
		# 4. GT global text semantic segmentaion map
		inputs['input_gt_global_masks'] = tf.placeholder(tf.uint8,\
							shape=[None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],\
							name='input_gt_global_masks')
	return inputs


# *****************************************************************************
# 
# 							build backbone FPN and TCM
# 			
# *****************************************************************************
def _extra_conv_arg_scope_with_bn(weight_decay=0.00001, activation_fn=tf.nn.relu,\
								batch_norm_decay=0.997, batch_norm_epsilon=1e-5,\
								batch_norm_scale=True):
	batch_norm_params = {
	'decay': batch_norm_decay,
	'epsilon': batch_norm_epsilon,
	'scale': batch_norm_scale,
	'updates_collections': tf.GraphKeys.UPDATE_OPS,
	}

	with slim.arg_scope(
		[slim.conv2d],
		padding='SAME',
		weights_regularizer=slim.l2_regularizer(weight_decay),
		weights_initializer=slim.variance_scaling_initializer(),
		activation_fn=activation_fn,
		normalizer_fn=slim.batch_norm,
		normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc

def build_FPN(images, config, is_training, backbone='resnet50'):
	# images: [batch, h, w, channels]
	# Return: pyramid_feature Dict{P2, P3, P4, P5} of feature maps from different level of the
	# 		  feature pyramid. Each is [batch, height, width, channels]
	pyramid = {}
	# build backbone network
	with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=1e-5)):
		if backbone == "resnet50":
			logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
			pyramid['C2'] = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
			pyramid['C3'] = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']
			pyramid['C4'] = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
			pyramid['C5'] = end_points['resnet_v1_50/block4/unit_3/bottleneck_v1']
		elif backbone == "resnet101":
			logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')
			pyramid['C2'] = end_points['resnet_v1_101/block1/unit_2/bottleneck_v1']
			pyramid['C3'] = end_points['resnet_v1_101/block2/unit_3/bottleneck_v1']
			pyramid['C4'] = end_points['resnet_v1_101/block3/unit_22/bottleneck_v1']
			pyramid['C5'] = end_points['resnet_v1_101/block4/unit_3/bottleneck_v1']
		else:
			print("Unkown backbone : ", backbone)
	# build FPN
	pyramid_feature = {}
	arg_scope = _extra_conv_arg_scope_with_bn()
	with tf.variable_scope('FPN'):
		with slim.arg_scope(arg_scope):
			pyramid_feature['P5'] = slim.conv2d(pyramid['C5'], config.TOP_DOWN_PYRAMID_SIZE, 1)
			for i in range(4, 1, -1):
				upshape = tf.shape(pyramid['C%d' % i])
				u = tf.image.resize_bilinear(pyramid_feature['P%d' % (i+1)], \
					size = (upshape[1], upshape[2]))
				c = slim.conv2d(pyramid['C%d' % i], config.TOP_DOWN_PYRAMID_SIZE, 1)
				s = tf.add(c, u)
				pyramid_feature['P%d' % i] = slim.conv2d(s, config.TOP_DOWN_PYRAMID_SIZE, 3)
	return pyramid_feature

def TCM_module(input_feature, image_shape, scop, config):
	'''
	Build the TCM module of SPCNet
	The global text segmentation branch acts on each 
	stage of the FPN to generate a semantic segmentation map of the text.
	input_feature: [batch, height, width, channels]
	image_shape: [h, w]

	Return: global_text_seg: [batch, height, width, 2]
	'''
	arg_scope = [slim.conv2d]
	with slim.arg_scope(arg_scope, 
					activation_fn=None,
					normalizer_fn=None,
					weights_initializer=tf.truncated_normal_initializer(stddev=0.001)):
		with tf.variable_scope(scop + "/TCM_Module"):
			conv1 = slim.conv2d(input_feature, config.TOP_DOWN_PYRAMID_SIZE, [3,3])
			conv2 = slim.conv2d(conv1, config.TOP_DOWN_PYRAMID_SIZE, [3,3])
			# global text semantic sementation map [N, h, w, 2]
			conv3 = slim.conv2d(conv2, 2, [1,1], activation_fn=None)
			text_prob = tf.nn.softmax(conv3)
			# saliency map [N, h, w, 1]
			active = tf.exp(text_prob[:,:,:,1])
			broadcast = tf.broadcast_to(active, tf.shape(input_feature))
			mult = tf.multiply(input_feature, broadcast)
			global_text_seg = tf.image.resize_bilinear(conv3, size = (image_shape[0], image_shape[1]))
			output = tf.add(conv1, mult)
			return global_text_seg, output

def build_TCM(pyramid_feature, image_shape, config):
	# build the text context module
	# 
	gts = {}
	tcm_outputs = {}
	for i in range(5, 1, -1):
		scop = 'P%d' % i 
		gts[scop], tcm_outputs[scop]  = TCM_module(pyramid_feature[scop],\
															image_shape, scop, config)
	return gts, tcm_outputs

#********************************************************************
#
#				build RPN generate proposals
#				
#*******************************************************************		
def build_RPN(pyramid_feature, image_shape, anchors_num, is_training, config):
	'''
	This inputs are outputs of TCM model
	pyramid_feature Dict{P2, P3, P4, P5} of feature maps from different level of the
	feature pyramid. Each is [batch, height, width, channels]
	'''
	rpn_class_logits = {}
	rpn_probs = {}
	rpn_bboxes = {}
	arg_scope = _extra_conv_arg_scope_with_bn(activation_fn=None)
	with slim.arg_scope(arg_scope):
		with tf.variable_scope('rpn'):
			for i in range(2, 6, 1):
				p = 'P%d' % i # [P2, P3, P4, P5]
				share_map = pyramid_feature[p]
				'''	
				share_map = slim.conv2d(pyramid[p], 256, [3,3], stride=config.RPN_ANCHOR_STRIDE,\
					activation_fn=tf.nn.relu, scop='%s/share_map' % p)
				'''
				rpn_class_logits[p] = slim.conv2d(share_map, anchors_num * 2, [1, 1], stride=1,\
					activation_fn=None, \
					weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
				rpn_bboxes[p] = slim.conv2d(share_map, anchors_num * 4, [1, 1], stride=1, \
					activation_fn=None, \
					weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
			# gather all rois
			# shape: [Batch, anchors, 4]
			rpn_bboxes_merge = tf.concat(values=[tf.reshape(rpn_bboxes['P%d'%i], (config.IMAGES_PER_GPU, -1, 4)) \
											for i in range(2,6)], axis=1)
			# shape: [Batch, anchors, 2]
			rpn_class_logits_merge = tf.concat(values=[tf.reshape(rpn_class_logits['P%d'%i], (config.IMAGES_PER_GPU, -1, 2)) \
											for i in range(2, 6)], axis=1)
			rpn_probs_merge = tf.nn.softmax(rpn_class_logits_merge)
	return rpn_class_logits_merge, rpn_probs_merge, rpn_bboxes_merge

	
def generate_all_anchors(fpn_shapes, image_shape, config):
	'''
	generate anchor for pyramid feature maps
	'''
	anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, \
											 config.RPN_ANCHOR_RATIOS, \
											 fpn_shapes, \
											 config.BACKBONE_STRIDES, \
											 config.RPN_ANCHOR_STRIDE)
	# normalize coordinates 
	# numpy array [N, 4] 
	norm_anchors = utils.norm_boxes(anchors, image_shape)
	anchors_tensor = tf.convert_to_tensor(norm_anchors)
	# Duplicate across the batch dimension
	batch_anchors = tf.broadcast_to(anchors_tensor,\
					[config.IMAGES_PER_GPU, tf.shape(anchors_tensor)[0],tf.shape(anchors_tensor)[1]])
	return batch_anchors

def generate_proposal(rpn_prob, rpn_bbox, anchors, proposal_count, config):
	nms_thresh = config.RPN_NMS_THRESHOLD
	# Box Scores [Batch, num_rois, 1]
	scores = rpn_prob[:,:,1]
	# Box deltas [batch, num_rois, 4]
	deltas = rpn_bbox * np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])
	# Improve performance by trimming to top anchors by score
	# and doing the rest on the smaller subset.
	pre_nms_limit = tf.minimum(config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
	ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, \
						name="top_anchors").indices
	scores = utils.batch_slice([scores, ix], \
						lambda x,y : tf.gather(x, y), config.IMAGES_PER_GPU)
	deltas = utils.batch_slice([deltas, ix], \
						lambda x,y : tf.gather(x, y), config.IMAGES_PER_GPU)
	pre_nms_anchors = utils.batch_slice([anchors, ix], \
						lambda x,y : tf.gather(x, y), config.IMAGES_PER_GPU)
	# Apply deltas to anchors to get refined anchors.
	# [batch, N, (y1, x1, y2, x2)]
	boxes = utils.batch_slice([pre_nms_anchors, deltas],\
						lambda x, y: utils.apply_box_deltas_graph(x, y),\
						config.IMAGES_PER_GPU, names=["refined_anchors"])

	# Clip to image boundaries. Since we're in normalized coordinates,
	# clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
	window = np.array([0, 0, 1, 1], dtype=np.float32)
	boxes = utils.batch_slice(boxes,\
						lambda x: utils.clip_boxes_graph(x, window),\
						config.IMAGES_PER_GPU,names=["refined_anchors_clipped"])
	# Filter out small boxes
	# According to Xinlei Chen's paper, this reduces detection accuracy
	# for small objects, so we're skipping it.
	# Non-max suppression
	def nms(boxes, scores):
		indices = tf.image.non_max_suppression(boxes, scores, proposal_count, nms_thresh,\
						name='rpn_non_max_suppression')
		proposals = tf.gather(boxes, indices)
		#Pad if needed
		padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
		proposals = tf.pad(proposals, [(0, padding), (0,0)])
		return proposals
	proposals = utils.batch_slice([boxes, scores], nms, \
						config.IMAGES_PER_GPU)
	proposals = tf.reshape(proposals, (-1, proposal_count, 4))
	return proposals


# *************************************************************
# 
# 				generate mask-rcnn heads targets
# 				
# *************************************************************
def detect_target(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	"""
	Generates detection targets for one image. Subsamples proposals and
	generates target class IDs, bounding box deltas, and masks for each.

	Inputs:
	proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [MAX_GT_INSTANCES] int class IDs
	gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
	gt_masks: [MAX_GT_INSTANCES, height, width] of boolean type.

	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
	class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
	deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
	masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
		   boundaries and resized to neural network output size.

	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""
	# Assertions
	asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],\
				name = 'roi_assertion')]
	with tf.control_dependencies(asserts):
		proposals = tf.identity(proposals)

	# Remove zero padding
	proposals, _ = utils.trim_zeros_graph(proposals, name='trim_proposals')
	gt_boxes, non_zeros = utils.trim_zeros_graph(gt_boxes, name='trim_gt_boxes')
	gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,\
									name='trim_gt_class_ids')
	gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:,0], axis=2,\
						 name='trim_gt_masks')
	# Compute overlaps matrix [proposals, gt_boxes]
	overlaps = utils.overlaps_graph(proposals, gt_boxes)

	# Determin positive and negative ROIs
	roi_iou_max = tf.reduce_max(overlaps, axis=1)

	# 1. Positive ROIs are those with >= 0.5 IoU with a GT box
	positive_roi_bool = (roi_iou_max >= 0.5)
	positive_indices = tf.where(positive_roi_bool)[:,0]
	# 2. Negative ROIs are those with < 0.5 with every GT box.
	negative_indices = tf.where(roi_iou_max < 0.5)[:,0]
	# Subsample ROIs, Aim for 33% positive
	# Positive ROIs
	positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
	positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
	positive_count = tf.shape(positive_indices)[0]
	# Negative ROIs. Add enough to maintain positive:negative ratio
	r = 1.0 / config.ROI_POSITIVE_RATIO
	negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32)\
						- positive_count
	negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
	# Gather selected ROIs
	positive_rois = tf.gather(proposals, positive_indices)
	negative_rois = tf.gather(proposals, negative_indices)

	# Assign positive ROIs to GT boxes
	positive_overlaps = tf.gather(overlaps, positive_indices)
	roi_gt_box_assignment = tf.cond(\
						tf.greater(tf.shape(positive_overlaps)[1], 0),\
						true_fn = lambda : tf.argmax(positive_overlaps, axis=1),\
						false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
						)
	roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
	roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
	# Compute bbox refinement for positive ROIs
	deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
	deltas /= config.BBOX_STD_DEV

	# Assign positive ROIs to GT masks
	# Permute masks to [N, height, width, 1]
	transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
	# Pick the right mask for each ROI
	roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

	# Compute mask targets 
	boxes = positive_rois
	if config.USE_MINI_MASK:
		# Transform ROI coordinates from normalized image space 
		# to normalized mini_mask space
		y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
		gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
		gt_h = gt_y2 - gt_y1
		gt_w = gt_x2 - gt_x1
		y1 = (y1 - gt_y1) / gt_h
		x1 = (x1 - gt_x1) / gt_w
		y2 = (y2 - gt_y2) / gt_h
		x2 = (x2 - gt_x2) / gt_w
		boxes = tf.concat([y1, x1, y2, x2], 1)
	box_ids = tf.range(0, tf.shape(roi_masks)[0])
	masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,\
									 box_ids,\
									 config.MASK_SHAPE)
	# Remove the extra dimension from masks
	masks = tf.squeeze(masks, axis=3)

	# Threshold mask pixels at 0.5 to have GT masks be 0 or 1\
	# to use with binary cross entropy loss
	masks = tf.round(masks)

	# Append negative ROIs and pad bbox deltas and masks that
	# are not used for negative ROIs with zeros.
	# positive_rois = tf.Print(positive_rois, ['positive_rois: ', tf.shape(positive_rois), "negative_rois: ", tf.shape(negative_rois)])
	rois = tf.concat([positive_rois, negative_rois], axis=0)
	N = tf.shape(negative_rois)[0]
	P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
	rois = tf.pad(rois, [(0,P), (0,0)])
	roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
	roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N+P)])
	deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
	masks = tf.pad(masks, [(0, N + P), (0, 0), (0, 0)])

	return rois, roi_gt_class_ids, deltas, masks

def generate_detect_target(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	"""
	Subsamples proposals and generates target box refinement, class_ids,
	and masks for each.

	Inputs:
	proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
	gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
			  coordinates.
	gt_masks: [batch, MAX_GT_INSTANCES, height, width] of boolean type

	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
		  coordinates
	target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
	target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
	target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
				 Masks cropped to bbox boundaries and resized to neural
				 network output size.

	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""
	# Slice the batch and run a graph for each slice
	names = ['rois', 'target_class_ids', 'target_deltas', 'target_mask']
	outputs = utils.batch_slice(\
				[proposals, gt_class_ids, gt_boxes, gt_masks],\
				lambda w,x,y,z,config : detect_target(w, x, y, z,config),\
				config.IMAGES_PER_GPU, names=names,config=config)
	return outputs

# ******************************************************************
# 
# 				build ROIAlign layer mrcnn head and get outputs
# 				
# ******************************************************************  

def PyramidROIAlign(boxes, image_shape, pyramid_feature, pool_shape, config):
	'''
	boxes: Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords

	pyramid_feature Dict{P2, P3, P4, P5} of feature maps from different level of the
	feature pyramid. Each is [batch, height, width, channels]

	pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

	Note the implement of ROI Pooling is slightly different from the Keras version
	'''
	# Assign each ROI to a level in the pyramid based on the ROI area.
	y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
	h = y2 - y1
	w = x2 - x1

	# Equation 1 in the Feature Pyramid Networks paper. Account for
	# the fact that our coordinates are normalized here.
	# e.g. a 224x224 ROI (in pixels) maps to P4
	image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
	# Make sure roi_level is bettwen 2 and 5
	roi_level = utils.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
	roi_level = tf.minimum(5, tf.maximum(\
				2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
	# [batch, num_boxes]
	roi_level = tf.squeeze(roi_level, 2)

	# Loop through levels and apply ROI pooling to each. P2 to P5.
	croped = []
	box_to_level = []
	for level in range(2, 6):
		p = 'P%d' % level
		# [Batch_index, boxes_index]
		ix = tf.where(tf.equal(roi_level, level))
		level_boxes = tf.gather_nd(boxes, ix)

		#Box indices for crop_and_resize
		box_indices = tf.cast(ix[:,0], tf.int32)

		# Keep track of which box is mapped to which level
		box_to_level.append(ix)

		#Stop gradient propogation to ROI proposals
		level_boxes = tf.stop_gradient(level_boxes)
		box_indices = tf.stop_gradient(box_indices)
		# Crop and Resize
		# Inputs params shape:
		# pyramid_feature: [batch, feature_map_height, feature_map_width, channels]
		# level_boxes: [this_level_num_boxes, 4]
		# box_indices: [this_level_num_boxes]
		# pool_shape: [pool_height, pool_width]
		# Result: [batch * num_boxes, pool_height, pool_width, channels]
		croped.append(tf.image.crop_and_resize(\
			pyramid_feature[p], level_boxes, box_indices, pool_shape,\
			method='bilinear'))
	# Pack croped features into one tensor
	croped = tf.concat(croped, axis=0)
	# Pack box_to_level mapping into one array and add another
	# Column representing the order of croped boxes
	box_to_level = tf.concat(box_to_level, axis=0)
	box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
	box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],axis=1)
	# Rearrange croped features to match the order of the original boxes
	# Sort box_to_level by batch_index then box index
	# TF doesn't have a way to sort by two columns, so merge them and sort.
	sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:,1]
	ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
	ix = tf.gather(box_to_level[:,2], ix)
	croped = tf.gather(croped, ix)

	# Re-add the batch dimention
	# croped regions in the shape: [batch * num_boxes, pool_height, pool_width, channels].
	# The width and height are those specific in the pool_shape in the layer constructor.
	shape = tf.concat([[tf.shape(boxes)[0] * tf.shape(boxes)[1]], tf.shape(croped)[1:]], axis=0)
	pooled = tf.reshape(croped, shape)
	return pooled

def build_mrcnn_head(rois, pyramid_feature, image_shape, is_training, config):
	# ROI Pooling
	# rois: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
	# pyramid_feature: Dict{P2, P3, P4, P5} of feature maps from different level of the
	# feature pyramid. Each is [batch, height, width, channels]
	# image_shape: [h, w]
	# 
	# return: mrcnn_class_logits: [batch, num_rois, num_classes]
	# 		  mrcnn_porbs: [batch, num_rois, num_classes]
	# 		  mrcnn_bbox: [batch, num_rois, num_classes * 4]
	arg_scope = _extra_conv_arg_scope_with_bn()
	with slim.arg_scope(arg_scope):
		with tf.variable_scope('mrcnn_head'):
			# ROIAlign and ROI max_pooling, shape [Batch_size * num_rois, pool_width, pool_height, channels]
			pooled_rois =  PyramidROIAlign(rois, image_shape, pyramid_feature, 
							[config.POOL_SIZE, config.POOL_SIZE], config)
			# Replace the full connection layer of fast rcnn with 1*1 conv
			conv1 = slim.conv2d(pooled_rois, config.FPN_CLASSIF_FC_LAYERS_SIZE, [1,1])
			conv2 = slim.conv2d(conv1, config.FPN_CLASSIF_FC_LAYERS_SIZE, [1,1])
			# [N, pool_w * pool_h * fc_layers_size]
			flatten = slim.flatten(conv2)
			# Classifier head
			mrcnn_class_logits = slim.fully_connected(flatten, config.NUM_CLASSES,\
								 activation_fn=None,\
								 weights_initializer=tf.truncated_normal_initializer(stddev=0.05))
			mrcnn_prob = tf.nn.softmax(mrcnn_class_logits)
			# BBox head
			# [batch * num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
			mrcnn_bbox = slim.fully_connected(flatten, config.NUM_CLASSES * 4, \
								 activation_fn=None,\
								 weights_initializer=tf.truncated_normal_initializer(stddev=0.05))
			# Reshape [Batch, num_rois, num_classes]
			cls_shape = tf.concat([tf.shape(rois)[0:2], tf.shape(mrcnn_class_logits)[1:]], axis=0)
			mrcnn_class_logits = tf.reshape(mrcnn_class_logits, cls_shape)
			mrcnn_prob = tf.reshape(mrcnn_prob, cls_shape)
			# Reshape [Batch, num_rois, num_classes * 4]
			box_shape = tf.concat([tf.shape(rois)[0:2], tf.shape(mrcnn_bbox)[1:]], axis=0)
			mrcnn_bbox = tf.reshape(mrcnn_bbox, box_shape)

			return mrcnn_class_logits, mrcnn_prob, mrcnn_bbox

def build_mrcnn_mask(rois, feature_maps, image_shape, is_training, config):
	"""
	Builds mask-rcnn mask head.

	rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
		  coordinates.
	feature_maps: List of feature maps from different layers of the pyramid,
				  [P2, P3, P4, P5]. Each has a different resolution.

	Returns:Mask_logits [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES] 
			Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]

	"""
	arg_scope = _extra_conv_arg_scope_with_bn()
	with slim.arg_scope(arg_scope):
		with tf.variable_scope('mrcnn_mask'):
			# ROI Pooling
			# Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
			pooled_rois = PyramidROIAlign(rois, image_shape, feature_maps,\
										  [config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],\
										  config)
			for i in range(4):
				pooled_rois = slim.conv2d(pooled_rois, 256, [3,3],\
										stride=1, padding='SAME')
			# 28 x 28
			conv = slim.conv2d_transpose(pooled_rois, 256, [2, 2],\
										stride=2, padding='VALID', activation_fn=tf.nn.relu)
			mask_logits = slim.conv2d(conv, config.NUM_CLASSES, [1,1],\
										stride=1, padding='VALID', activation_fn=None)
			mask_shape = tf.concat([tf.shape(rois)[0:2], tf.shape(mask_logits)[1:]], axis=0)
			mrcnn_mask_logits = tf.reshape(mask_logits, mask_shape)
			mrcnn_mask_pred = tf.sigmoid(mrcnn_mask_logits)

			return mrcnn_mask_logits, mrcnn_mask_pred

def get_detect_results(rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, gts, config):
	# Run detection refinement graph on each item in the batch
	detections_batch, mask_batch = utils.batch_slice(\
		[rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, gts],\
		lambda x, y, z, m, g, config : refine_detections_graph(\
			x, y, z, m, g, config), config.IMAGES_PER_GPU, config=config)
	# Reshape output
	# [batch, config.DETECTION_MAX_INSTANCES, (y1, x1, y2, x2, class_id, score)]
	# where coordinates are normalized
	detections_batch = tf.reshape(detections_batch, \
		[tf.shape(rois)[0], config.DETECTION_MAX_INSTANCES, 6])
	# [batch, config.DETECTION_MAX_INSTANCES, MASK_H, MASK_W, CLASS_NUM]
	mask_batch = tf.reshape(mask_batch, \
		[tf.shape(rois)[0], config.DETECTION_MAX_INSTANCES, \
			config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES])
	return detections_batch, mask_batch

def refine_detections_graph(rois, probs, deltas, mrcnn_mask, gts, config=None):
	"""
	Refine classified proposals and filter overlaps and return final
	detections.

	Inputs:
		rois: [N, (y1, x1, y2, x2)] in normalized coordinates
		probs: [N, num_classes]. Class probabilities.
		deltas: [N, num_classes * 4]. Class-specific
				bounding box deltas.
		mrcnn_mask: [N, MASK_H, MASK_W, NUM_CLASSES]
		gts: [4, H, W, NUM_CLASSES]
	
	Returns detections shaped: [config.DETECTION_MAX_INSTANCES, (y1, x1, y2, x2, class_id, score)] where
		coordinates are normalized.
		masks: [config.DETECTION_MAX_INSTANCES, MASK_H, MASK_W, 2] with zero padding
	"""
	# Reshape: [N, num_classes, (dy, dx, log(dh), log(dw))]
	deltas = tf.reshape(deltas, [tf.shape(deltas)[0], config.NUM_CLASSES, 4])
	# Class IDs per ROI
	class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
	# Class probability of the top class of each ROI
	indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
	class_scores = tf.gather_nd(probs, indices)
	# Class-specific bounding bpx deltas
	deltas_specific = tf.gather_nd(deltas, indices)
	# Apply bounding box deltas
	# Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
	refined_rois = utils.apply_box_deltas_graph(
		rois, deltas_specific * config.BBOX_STD_DEV)
	# Clip boxes to image shape
	refined_rois = utils.clip_boxes_graph(refined_rois,\
				tf.constant([0., 0., 1., 1.]))

	# class_scores = tf.Print(class_scores, ['orignal scores',class_scores],summarize=100)
	class_scores = build_rescore_graph(refined_rois, class_scores, \
					mrcnn_mask, gts, config)
	
	# class_scores = tf.Print(class_scores, ['rescores :', class_scores], summarize=100)
	# Filter out background boxes
	keep = tf.where(class_ids > 0)[:, 0]
	# Filter out low confidence boxes
	if config.DETECTION_MIN_CONFIDENCE:
		conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
		keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
										tf.expand_dims(conf_keep, 0))
		keep = tf.sparse_tensor_to_dense(keep)[0]

	# Apply per-class NMS
	# 1. Prepare variables
	pre_nms_class_ids = tf.gather(class_ids, keep)
	pre_nms_scores = tf.gather(class_scores, keep)
	pre_nms_rois = tf.gather(refined_rois,   keep)
	unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

	def nms_keep_map(class_id):
		"""Apply Non-Maximum Suppression on ROIs of the given class."""
		# Indices of ROIs of the given class
		ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
		# Apply NMS
		class_keep = tf.image.non_max_suppression(
				tf.gather(pre_nms_rois, ixs),
				tf.gather(pre_nms_scores, ixs),
				max_output_size=config.DETECTION_MAX_INSTANCES,
				iou_threshold=config.DETECTION_NMS_THRESHOLD)
		# Map indices
		class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
		# Pad with -1 so returned tensors have the same shape
		gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
		class_keep = tf.pad(class_keep, [(0, gap)],
					mode='CONSTANT', constant_values=-1)
		# Set shape so map_fn() can infer result shape
		class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
		return class_keep

	# 2. Map over class IDs
	nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
						 dtype=tf.int64)
	# 3. Merge results into one list, and remove -1 padding
	nms_keep = tf.reshape(nms_keep, [-1])
	nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
	# 4. Compute intersection between keep and nms_keep
	keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
									tf.expand_dims(nms_keep, 0))
	keep = tf.sparse_tensor_to_dense(keep)[0]
	# Keep top detections
	roi_count = config.DETECTION_MAX_INSTANCES
	class_scores_keep = tf.gather(class_scores, keep)
	num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
	top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
	keep = tf.gather(keep, top_ids)

	# Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
	# Coordinates are normalized.
	detections = tf.concat([
		tf.gather(refined_rois, keep),
		tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
		tf.gather(class_scores, keep)[..., tf.newaxis]
		], axis=1)
	masks = tf.gather(mrcnn_mask, keep)
	# Pad with zeros if detections < DETECTION_MAX_INSTANCES
	gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
	detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
	# Pad the masks with zeros
	mask_gap = config.DETECTION_MAX_INSTANCES - tf.shape(masks)[0]
	masks = tf.pad(masks, [(0, mask_gap), (0, 0), (0, 0), (0,0)], 'CONSTANT')
	return detections, masks

def build_rescore_graph(refined_rois, class_scores, mrcnn_mask, gts, config):
	"""Build rescore mechanism.
	refined_rois: [boxes_num, (y1, x1, y2, x2)] in normalized coordinates
	class_scores: [boxes_num, 1]
	mrcnn_mask:   [boxes_num, MASK_HEGIHT, MASK_WDITH, 2]
	gts: global text segmentation map.[4, H, W, 2].axis=0 represent P2, P3, P4, P5

	Return: [boxes_num, 1]
	"""
	# Make sure roi_level is bettwen 0 and 3: [boxes_num, 1]
	# Get the feature pyramid level corresponding to each box
	y1, x1, y2, x2 = tf.split(refined_rois, 4, axis=1)
	h = y2 - y1
	w = x2 - x1
	box_level = utils.log2_graph(tf.sqrt(h * w) / 224.0)
	box_level = tf.minimum(3, tf.maximum(\
				0, 2 + tf.cast(tf.round(box_level), tf.int32)))
	# [boxes_num, MASK_H, MASK_W, 2]
	box_gts = tf.image.crop_and_resize(gts, refined_rois,
				box_level[:,0], config.MASK_SHAPE)
	# bool mask: [boxes_num, MASK_H, MASK_W]
	mrcnn_mask = tf.greater(mrcnn_mask[:,:,:,1], tf.constant(0.5))
	# IS: [boxes_num, 1]
	instance_score_0 = tf.boolean_mask(box_gts[:,:,:,0], mrcnn_mask)
	instance_score_0 = tf.reduce_mean(instance_score_0, axis=0)
	instance_score_1 = tf.boolean_mask(box_gts[:,:,:,1], mrcnn_mask)
	instance_score_1 = tf.reduce_mean(instance_score_1, axis=0)
	# CS
	classify_score_0 = tf.constant(1.0) - class_scores
	classify_score_1 = class_scores
	# calculate fused score
	fused_score = tf.exp(instance_score_1 + classify_score_1)
	factor = tf.exp(instance_score_0 + classify_score_0) + fused_score
	box_rescore = fused_score / factor
	return box_rescore



#*****************************************************
#
#			calculate Losses
#			
#*****************************************************

def smooth_l1_loss(y_true, y_pred):
	"""
	Implements Smooth-L1 loss.
	y_true and y_pred are typically: [N, 4], but could be any shape.
	"""
	diff = tf.abs(y_true - y_pred)
	less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
	loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
	return loss

def build_rpn_class_loss(rpn_match, rpn_class_logits, config):
	'''
	rpn_match: [batch, anchors, 1] Anchor match type.
	1 = positive, -1 = negative, 0 = neutral anchor
	rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG
	'''
	# Squeeze last dim to simplify
	rpn_match = tf.squeeze(rpn_match, -1)
	# Get anchor classes. Convert the -1/+1 match 0/1 values
	anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
	# Positive and Negative anchors contribute to the loss,
	# but neutral anchors (match value = 0) don't
	indices = tf.where(tf.not_equal(rpn_match, 0))
	# Pick rows that contribute to the loss
	# and filter out the rest
	rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
	anchor_class = tf.gather_nd(anchor_class, indices)
	# Cross entropy loss
	# one-hot encoding anchor_class reshape to [N, 2]
	anchor_class = tf.one_hot(tf.reshape(anchor_class, [-1]),\
							depth=config.NUM_CLASSES,\
							on_value=1.0, off_value=0.0)
	rpn_class_logits = tf.reshape(rpn_class_logits, [-1,config.NUM_CLASSES])
	loss = tf.nn.softmax_cross_entropy_with_logits(\
					labels=anchor_class,\
					logits=rpn_class_logits)
	loss = tf.reduce_mean(loss)
	return loss

def build_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox, config):
	"""
	Return the RPN bounding box loss graph.

	target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
		Uses 0 padding to fill in unsed bbox deltas.
	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
	"""
	# Positive anchors contribute to the loss, but negative and
	# neutral anchors (match value of 0 or -1) don't.
	rpn_match = tf.squeeze(rpn_match, -1)
	indices = tf.where(tf.equal(rpn_match, 1))

	# Pick bbox deltas that contribute to the loss
	rpn_bbox = tf.gather_nd(rpn_bbox, indices)
	# Trim target bounding box deltas to the same length as rpn_bbox
	batch_counts = tf.reduce_sum(tf.cast(\
				tf.equal(rpn_match, 1), tf.int32), axis=1)
	target_bbox = utils.batch_pack_graph(target_bbox, batch_counts,\
				config.IMAGES_PER_GPU)
	# input:[N,4] output: [N, 1]
	loss = tf.cond(tf.size(target_bbox) > 0, 
		lambda: smooth_l1_loss(target_bbox, rpn_bbox),
		lambda: tf.constant(0.0))
	loss = tf.reduce_mean(loss)
	return loss

def build_mrcnn_class_loss(target_class_ids, pred_class_logits, config):
	"""
	Loss for the classifier head of Mask RCNN.

	target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
		padding to fill in the array.
	pred_class_logits: [batch, num_rois, num_classes]
	"""
	target_class_ids = tf.one_hot(\
						tf.reshape(target_class_ids, [-1]),\
						depth=config.NUM_CLASSES,\
						on_value=1.0, off_value=0.0)
	pred_class_logits = tf.reshape(pred_class_logits, [-1,config.NUM_CLASSES])
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_class_ids,\
							logits=pred_class_logits)
	loss = tf.reduce_mean(loss)
	return loss

def build_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox, config):
	"""
	Loss for Mask R-CNN bounding box refinement.

	target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
	target_class_ids: [batch, num_rois]. Integer class IDs.
	pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
	"""
	# Reshape to merge batch and roi dimensions for simplicity.
	target_class_ids = tf.reshape(target_class_ids, (-1,))
	target_bbox = tf.reshape(target_bbox, (-1, 4))
	pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))
	# Only positive ROIs contribute to the loss. And only
	# the right class_id of each ROI. Get their indices.
	positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
	positive_roi_class_ids = tf.cast(\
				tf.gather(target_class_ids, positive_roi_ix),\
				tf.int64)
	indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

	# Gather the deltas (predicted and true) that contribute to loss
	target_bbox = tf.gather(target_bbox, positive_roi_ix)
	pred_bbox = tf.gather_nd(pred_bbox, indices)

	#Smooth-L1 loss
	loss = tf.cond(tf.size(target_bbox) > 0, 
		lambda: smooth_l1_loss(target_bbox, pred_bbox),
		lambda: tf.constant(0.0))
	loss = tf.reduce_mean(loss)
	return loss

def build_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks, config):
	"""
	Mask binary cross-entropy loss for the masks head.

	target_masks: [batch, num_rois, height, width].
		A float32 tensor of values 0 or 1. Uses zero padding to fill array.
	target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
	pred_masks: [batch, proposals, height, width, num_classes] float32 tensor logits.
	"""
	# Reshape for simplicity. Merge first two dimensions into one.
	target_class_ids = tf.reshape(target_class_ids, (-1,))
	mask_shape = tf.shape(target_masks)
	target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
	pred_shape = tf.shape(pred_masks)
	pred_masks = tf.reshape(pred_masks,\
				(-1, pred_shape[2], pred_shape[3], pred_shape[4]))
	# Permute predicted masks to [N, num_classes, height, width]
	pred_masks = tf.transpose(pred_masks, [0, 3, 2, 1])
	# Only positive ROIs contribute to the loss. And only
	# the class specific mask of each ROI.
	positive_ix = tf.where(target_class_ids > 0)[:, 0]
	positive_class_ids = tf.cast(\
				tf.gather(target_class_ids, positive_ix), tf.int64)
	indices = tf.stack([positive_ix, positive_class_ids], axis=1)

	# Gather the masks (predicted and true) that contribute to loss
	y_true = tf.gather(target_masks, positive_ix)
	y_pred = tf.gather_nd(pred_masks, indices)

	# Compute binary cross entropy.
	# shape: [batch, roi, num_classes]
	loss = tf.cond(tf.size(y_true) > 0,
		lambda: tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred),
		lambda: tf.constant(0.0))
	loss = tf.reduce_mean(loss)
	return loss

def build_global_mask_loss(input_global_masks, pred_global_masks,config):
	"""
	Compute gloable text segmentation mask loss
	input_global_masks: [batch_size, height, width]
	pred_global_masks: dict{p2, p3, p4, p5}
						dict[p_n]: [batch_size, height, width, 2]
	"""
	losses = []
	# One hot encoding: [batch_size * height * width, 2]
	input_global_masks = tf.one_hot(
						tf.reshape(input_global_masks, [-1]),
						depth=config.NUM_CLASSES,
						on_value=1.0, off_value=0.0)
	for i in range(2, 6):
		p = "P%d" % i
		# [batch_size * h * w, 2] 
		pred_masks = tf.reshape(pred_global_masks[p], [-1, config.NUM_CLASSES])
		loss = tf.cond(tf.size(input_global_masks) > 0,
					lambda: tf.nn.sigmoid_cross_entropy_with_logits(labels=input_global_masks, logits=pred_masks),
					lambda: tf.constant(0.0))
		loss = tf.reduce_mean(loss)
		losses.append(loss)
	losses = tf.convert_to_tensor(losses)
	loss = tf.reduce_mean(losses)
	return loss
